#include <cassert>
#include <string>

#include "../barrier_id_map.h"

int main() {
  assert(trace_bar_id_for_op_bar("BAR.SYNC.DEFER_BLOCKING", 0x1df0, 64, NULL) ==
         0);
  assert(trace_bar_id_for_op_bar("BAR.SYNC.DEFER_BLOCKING", 0x2270, 64, NULL) ==
         0);

  trace_barrier_id_map map;
  unsigned nondefault_a =
      trace_bar_id_for_op_bar("BAR.SYNC", 0x3000, 64, &map);
  unsigned nondefault_b =
      trace_bar_id_for_op_bar("BAR.SYNC", 0x3010, 64, &map);
  assert(nondefault_a != nondefault_b);
  assert(nondefault_a == trace_bar_id_for_op_bar("BAR.SYNC", 0x3000, 64, &map));

  return 0;
}
