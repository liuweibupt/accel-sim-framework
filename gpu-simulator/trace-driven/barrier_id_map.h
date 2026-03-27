#ifndef ACCELSIM_TRACE_BARRIER_ID_MAP_H
#define ACCELSIM_TRACE_BARRIER_ID_MAP_H

#include <string>
#include <stdexcept>
#include <unordered_map>

class trace_barrier_id_map {
 public:
  unsigned get_or_assign(unsigned pc, unsigned max_slots) {
    std::unordered_map<unsigned, unsigned>::const_iterator it =
        m_pc_to_id.find(pc);
    if (it != m_pc_to_id.end()) return it->second;

    unsigned next_id = m_pc_to_id.size();
    if (next_id >= max_slots) {
      throw std::runtime_error(
          "trace_barrier_id_map exhausted configured barrier slots");
    }

    m_pc_to_id[pc] = next_id;
    return next_id;
  }

  size_t size() const { return m_pc_to_id.size(); }

 private:
  std::unordered_map<unsigned, unsigned> m_pc_to_id;
};

inline unsigned trace_bar_id_for_op_bar(const std::string &opcode,
                                        unsigned pc, unsigned max_slots,
                                        trace_barrier_id_map *map) {
  if (opcode == "BAR.SYNC.DEFER_BLOCKING") {
    return 0;
  }

  if (map == NULL) {
    throw std::runtime_error("trace_bar_id_for_op_bar requires map");
  }

  return map->get_or_assign(pc, max_slots);
}

#endif
