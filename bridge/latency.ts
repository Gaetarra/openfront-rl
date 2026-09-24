import { PseudoRandom } from "../OpenFront_Source_Code/openfrontio/src/core/PseudoRandom";
import { StampedIntent } from "../OpenFront_Source_Code/openfrontio/src/core/Schemas";

/**
 * Delays agent intents like a real client's network latency: L = 1, 2 or 3 turns with p = .6/.3/.1 (PLAN decision
 * 12). Intents that fall due in the same turn are shuffled so no seat always acts first.
 */
export class LatencyQueue {
  private pending: { due: number; intent: StampedIntent }[] = [];

  constructor(private readonly random: PseudoRandom) {}

  /** Queues an intent decided before turn `nextTurn` executes. */
  push(intent: StampedIntent, nextTurn: number): void {
    const r = this.random.next();
    this.pending.push({ due: nextTurn + (r < 0.6 ? 0 : r < 0.9 ? 1 : 2), intent });
  }

  /** The intents to execute in `turn`, in shuffled order. */
  take(turn: number): StampedIntent[] {
    const due = this.pending.filter((p) => p.due <= turn).map((p) => p.intent);
    if (due.length === 0) return due;
    this.pending = this.pending.filter((p) => p.due > turn);
    return due.length > 1 ? this.random.shuffleArray(due) : due;
  }
}
