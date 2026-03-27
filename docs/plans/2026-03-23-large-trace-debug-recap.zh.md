# 大 Trace Replay 调试复盘

## 范围
本文档记录当前 Accel-Sim 中大规模 A100 风格 trace replay 的调试状态，当前聚焦的负载是：

- 形状：`2048 x 12288 x 12288`
- trace 来源：Modal A100 重采样
- replay 输入：重新处理后的 `traceg`
- 目标配置：`SM80_A100`

## 目标
让这份大 GEMM trace 能在 Accel-Sim 中成功 replay，并把剩余 deadlock 的根因定位清楚，直到足以支撑修复。

---

## 最新结论（2026-03-27）

这份大 trace 现在已经可以在 `SM80_A100` 配置下成功 replay 结束，不再停在之前稳定复现的 deadlock 点。

成功产物目录：

- `modal/artifacts/fp16-2048x12288x12288-gemm3-retrace3/reprocessed_sim_bar0_20260327-010117_pty`

成功 replay 的关键统计：

- `gpu_tot_sim_cycle = 4848187`
- `gpu_tot_sim_insn = 10244247315`
- `gpu_tot_ipc = 2113.0059`
- `gpgpu_simulation_time = 0 days, 2 hrs, 29 min, 13 sec (8953 sec)`
- 日志结束标记：`GPGPU-Sim: *** exit detected ***`

这说明当前这条大 GEMM replay 链路已经跑通。

同时，这也把当前主根因判断进一步收敛为：

> trace-driven replay 对 `BAR.SYNC.DEFER_BLOCKING` 的 barrier 语义建模不正确，导致同一个 CTA 内的 warp 被错误拆分到不同 barrier phase 上，最终互相等待。

当前实验性但有效的修复方式是：

- 对 `BAR.SYNC.DEFER_BLOCKING` 统一映射到默认 CTA barrier 0
- 其它 `OP_BAR` 形式仍保留按 kernel 内 barrier PC 分配 id 的逻辑

也就是说，之前怀疑的 zero-mask `LDG.E` / `LDGSTS` / scheduler forward progress 问题，至少对这份已经重新处理好的大 trace 而言，不再是主阻塞项。

---

## 已经解决的问题

### 1. 最早的大 trace 本身不完整
最早的一版大 trace replay 失败，根因不是模拟器，而是 trace 自己不完整：

- grid 预计应该有 `1536` 个 CTA
- 实际 trace 里只记录到了 `569` 个 thread block 结束

这份 trace 后来已经废弃，换成了重新生成的完整 trace。

### 2. `post-traces-processing` 把正确的原始 trace 处理坏了
原始 `.trace` 文件本身结构是对的，但生成出的 `.traceg` 丢了某些 warp 记录。

一个明确例子：

- `thread block = 53,4,0`
- 原始 `.traceg` 里缺少 `warp = 5`
- 但原始 `.trace` 里仍然有对应数据

根因：

- `post-traces-processing.cpp` 里复用 `stringstream` 时没有 `ss.clear()`

当前状态：

- 已修复
- 原始 trace 已重新本地处理
- 新生成的 `traceg` 已完成结构校验

### 3. trace-driven 的 `OP_BAR` 把所有 barrier phase 都压成了一个
原始 trace-driven 解码里：

```cpp
bar_id = 0;
```

对于 SM80 这类 software-pipelined GEMM，这个语义太粗糙，会把不同 phase 的 barrier 全部混到一个 barrier 上。

### 4. 用 PC hash 的 barrier id 仍然不够
第一轮修复把 `bar_id` 改成了 PC 派生 hash，但 `% MAX_BARRIERS_PER_CTA` 仍然会发生碰撞。

这对大 trace 依然不够。

### 5. 已实现按 kernel 内 barrier PC 精确分配 id
当前 trace-driven barrier 映射逻辑已经变成：

- 每个 kernel 内，每个唯一 barrier PC 分配一个唯一 barrier id
- 相同 barrier PC 重复出现时复用同一个 id
- 如果 barrier slot 数量不够，会显式报错，而不是静默冲突

当前状态：

- 已实现
- 已做回归测试
- 已提交并 push

### 5.1 `BAR.SYNC.DEFER_BLOCKING` 不能简单按 barrier PC 拆 phase
继续调试大 trace 后发现，上一条“按 barrier PC 精确分配 id”的修复还不够。

原因是：

- 大型 CUTLASS GEMM 中会反复出现多个 `BAR.SYNC.DEFER_BLOCKING` PC
- 它们在 trace 中并没有暴露出可直接区分的显式 barrier operand
- 如果直接按 PC 给它们不同 id，会把同一个 CTA 的 warp 错误拆到不同 barrier phase

在 deadlock dump 中，可以明确看到 warp 被分裂卡在：

- `bar_id=2, bar_pc=0x1df0`
- `bar_id=3, bar_pc=0x2270`

而这两个位置对应的都是 `BAR.SYNC.DEFER_BLOCKING`。

因此当前收敛出的更合理语义是：

- `BAR.SYNC.DEFER_BLOCKING` → 统一映射到默认 barrier 0
- 其它 `OP_BAR` → 仍使用 per-kernel PC map

这也是最终把大 replay 跑通的关键修复。

### 6. empty warp trace 生命周期 bug
之前 replay 曾经失败在：

```text
trace_driven.cc:82: assert(warp_traces.size() > 0)
```

这表示空 trace warp 被错误地继续当作正常 warp 使用了。

当前状态：

- 已修复为空 trace warp 直接完成并 `warp_exit`

### 7. warp 退出时 barrier 清理问题
之前还暴露过一个问题：最后一个 active warp 退出时，barrier 状态没有清干净。

当前状态：

- 已修复，在 active warp 集合为空时也能正确释放 barrier waiter

### 8. 本地 standalone binary / ABI mismatch
在前几轮修复之后，本地 replay binary 仍然会 startup crash，原因是：

- 一部分 simulator object 还是旧布局编出来的
- 另一部分 object 是在 `MAX_BARRIERS_PER_CTA` 从 `16` 改到 `64` 之后重新编的

这会导致混合 ABI 重链接，从而引发启动阶段崩溃。

当前状态：

- 相关 simulator object 已重编
- standalone relink 已修复
- small replay 已恢复

### 9. small replay 已恢复健康
已验证成功的 smoke replay：

- FP16 `512x512x512`

观测到的成功输出：

- `gpu_tot_sim_cycle = 35980`
- `gpu_tot_sim_insn = 6150144`
- `gpu_tot_ipc = 170.9323`
- `GPGPU-Sim: *** exit detected ***`

---

## 之前大 trace 的现象（现已被修复）

现在的大 replay 不再是“一启动就坏”，而是已经能稳定跑到较深阶段，然后复现出一个稳定的后期 deadlock。

稳定 deadlock 特征如下：

```text
GPGPU-Sim uArch: ERROR ** deadlock detected:
last writeback core 35 @ gpu_sim_cycle 7689340
(+ gpu_tot_sim_cycle 4287217296) (60660 cycles ago)
```

这个 deadlock 点在旧语义下可以稳定复现，但在当前 `BAR.SYNC.DEFER_BLOCKING -> barrier 0` 的修复语义下，已经不再出现。

---

## 之前 deadlock 的关键观察

### 1. CTA 0 的 barrier 清理不是最终根因
在 deadlock 日志末尾可以看到：

- CTA 0 的 barrier phase 完整结束
- `warp_exit_*` 完整结束
- `deallocate_pre` 显示 CTA teardown 是干净的

因此，当前剩余 deadlock **不是**“CTA 0 的 barrier 最后没放掉”。

### 2. core 35 不是实际卡住的核心
更细的 gdb pipeline dump 显示：

- core `35` 在 deadlock 时是 `0 threads running`
- pipeline 全空 / 全 bubble

因此：

- core 35 只是 **最后一个 writeback 的 core**
- 不是真正阻止 forward progress 的 core

### 3. 真正有问题的 core 仍然是活跃的
比如 core `0` 和 core `2` 的 dump 显示：

- `256 threads running`
- warp 还活着
- ibuffer 里还有指令
- pipeline / scoreboard / operand collector 基本都空
- 没有明显的 memory response backlog

这意味着：

- 一些 warp 还活着
- 但前端 / scheduler / issue 路径已经不再产生 forward progress

### 4. deadlock 时反复出现的关键 PC
在 deadlock 时观察到的 warp PC 主要包括：

- `0x1e00`
- `0x2280`

映射回 trace 指令后分别是：

- `0x1e00` -> `IADD3.X`
- `0x1e10` -> `HMMA.16816.F32`
- `0x2280` -> `LDG.E`，并且 **active mask = 00000000**
- `0x2290` -> `LDG.E`，并且 **active mask = 00000000**

这是当前最强的一条线索之一。

### 5. 为什么 zero-mask load 可疑
当前 deadlocked 的 warp 看上去仍然“活着”，而且 ibuffer 里也还有指令，但它们不再发出有效 issue。

其中 `LDG.E` 的 zero-mask 情况尤其可疑，因为它可能会错误影响：

- warp progress bookkeeping
- issue / completion 逻辑
- scoreboard 可见性
- 前端对“这个 warp 还有没有活”的判断

当前假设是：

- 当前 replay 已经不再主要受 barrier identity 崩坏支配
- 更可能的主问题是 **alive-but-not-advancing warp 的 forward-progress 逻辑**
- 而 zero-mask memory 指令是当前最可疑对象之一

---

## 已经做过的进一步调试

### gdb deadlock 抓取
我已经用脚本化 gdb replay 抓到过：

- deadlock 断点
- backtrace
- 指定 shader core 的 pipeline dump
- memory partition 0 dump

这些抓取已经确认了：

- deadlock 是真实的，而且稳定
- core 35 不是根 stuck core
- 活跃 core 上仍然有 live warp 和前端状态

### 已加入的 instrumentation
我已经在本地 nested `gpgpu-sim` 里加入了新的 deadlock 诊断 instrumentation：

- deadlock 时自动 dump 指定 core 的 pipeline
- 扩展 warp 状态打印，计划显示等待原因，例如：
  - `wait=barrier`
  - `wait=mem_barrier`
  - `wait=ldgsts`
  - `wait=atomic(...)`

这些 instrumentation 已经编入最新 standalone replay binary。

---

## 当前最合理的判断

整个 replay 链路最终走过了之前那些更早期的错误阶段：

- trace 结构损坏
- barrier identity collapse
- empty warp 生命周期错误
- ABI / relink 崩溃

最终事实表明，之前的死锁更接近于：

> `BAR.SYNC.DEFER_BLOCKING` 的 trace replay 语义错误，导致 warp barrier phase 被错误拆分。

而不是：

- 单纯的 zero-mask `LDG.E` 问题
- 单纯的 `LDGSTS` completion 问题
- 单纯的 scheduler forward-progress 问题

---

## 当前结论与后续方向

对这份大 trace 而言，当前已经拿到成功 replay，因此下一步重点不再是“先把它跑通”，而是：

1. 把当前有效修复整理为更干净的正式补丁
2. 判断这种 `DEFER_BLOCKING -> barrier 0` 的处理是否还能泛化到更多 SM80 trace
3. 用更多 CUTLASS / transformer GEMM trace 做回归
4. 再决定是否保留当前 nested `gpgpu-sim` 中的调试 instrumentation

---

## 分支 / 产物说明
本文档是当前状态复盘。

相关实现工作已经记录在分支：

- `modal-a100-cutlass-trace`

另外还有一些本地 nested 仓库里的 debug instrumentation 改动存在于：

- `gpu-simulator/gpgpu-sim`

后续真正收敛最终 fix 时，需要谨慎整理这些改动。
