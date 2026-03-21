#include <cassert>
#include <stdexcept>

#include "../barrier_id_map.h"

int main() {
  trace_barrier_id_map map;

  unsigned id_a = map.get_or_assign(0x1df0, 64);
  unsigned id_b = map.get_or_assign(0x71f0, 64);
  unsigned id_c = map.get_or_assign(0x5a30, 64);
  unsigned id_d = map.get_or_assign(0x6e30, 64);

  assert(id_a == map.get_or_assign(0x1df0, 64));
  assert(id_b == map.get_or_assign(0x71f0, 64));
  assert(id_a != id_b);
  assert(id_c != id_d);

  trace_barrier_id_map overflow;
  bool threw = false;
  try {
    overflow.get_or_assign(0x1000, 1);
    overflow.get_or_assign(0x2000, 1);
  } catch (const std::runtime_error &) {
    threw = true;
  }
  assert(threw);

  return 0;
}
