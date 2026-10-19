import assert from "node:assert/strict";
import { test } from "node:test";
import { Player } from "../OpenFront_Source_Code/openfrontio/packages/engine/src/game/Game";
import { Act, decodeAction, DIPLOMACY, Masks, seatMasks } from "./actions";
import { EnvGame } from "./game";
import { FFA, HVN, LONE, quickGame, spawnSeats } from "./testkit";

/** Whether `act` may target player `p`, as the mask says. */
const row = (m: Masks, act: Act, p: Player) => m.diplomacy[DIPLOMACY.indexOf(act) * (m.diplomacy.length / DIPLOMACY.length) + p.smallID()];
/** Observes, then steps with the given actions by seat and no-ops for the other seats. */
const act = (env: EnvGame, picks: Record<number, number[]>) => {
  env.observe(false);
  env.step(env.seats.map((_, i) => picks[i] ?? [Act.Noop, 0, 0, 0]));
};

test("a request is legal toward live non-friends, and a request back forms the alliance", async () => {
  const env = await quickGame(FFA, 5, LONE);
  spawnSeats(env);
  const [a, b] = env.seats;
  const m = seatMasks(env.game, a, env.slots);
  assert.deepEqual([row(m, Act.AllianceRequest, b), row(m, Act.AllianceRequest, a)], [1, 0]);
  assert.deepEqual([m.types[Act.AllianceRequest], m.types[Act.BreakAlliance]], [1, 0]);
  assert.deepEqual(decodeAction(env.game, a, env.frame, m, [Act.AllianceRequest, 0, b.smallID(), 0]).intent,
    { type: "allianceRequest", recipient: b.id() });
  act(env, { 0: [Act.AllianceRequest, 0, b.smallID(), 0] });
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.AllianceRequest, b), 0, "a request is pending");
  act(env, { 1: [Act.AllianceRequest, 0, a.smallID(), 0] });
  assert.ok(a.isAlliedWith(b));
  assert.deepEqual(env.invalid, [false, false, false, false]);
});

test("renewal opens in the alliance's last offer window and renews when both ask", async () => {
  const env = await quickGame(FFA, 5, { ...LONE, customAllianceDuration: 1 });
  spawnSeats(env);
  const [a, b] = env.seats;
  act(env, { 0: [Act.AllianceRequest, 0, b.smallID(), 0], 1: [Act.AllianceRequest, 0, a.smallID(), 0] });
  const alliance = a.allianceWith(b)!;
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.RenewAlliance, b), 0, "too early");
  const expires = alliance.expiresAt();
  while (env.game.ticks() < expires - env.game.config().allianceExtensionPromptOffset()) env.tickWith([]);
  assert.equal(row(seatMasks(env.game, a, env.slots), Act.RenewAlliance, b), 1);
  act(env, { 0: [Act.RenewAlliance, 0, b.smallID(), 0], 1: [Act.RenewAlliance, 0, a.smallID(), 0] });
  assert.ok(a.allianceWith(b)!.expiresAt() > expires);
});

test("breaking marks the seat a traitor; an alliance that runs out marks nobody", async () => {
  // Two alliances in one game: a breaks the one with b and lets the one with c expire.
  const env = await quickGame(FFA, 5, { ...LONE, customAllianceDuration: 1 });
  spawnSeats(env);
  const [a, b, c, d] = env.seats;
  act(env, { 0: [Act.AllianceRequest, 0, b.smallID(), 0], 1: [Act.AllianceRequest, 0, a.smallID(), 0] });
  act(env, { 0: [Act.AllianceRequest, 0, c.smallID(), 0], 2: [Act.AllianceRequest, 0, a.smallID(), 0] });
  assert.ok(a.isAlliedWith(b) && a.isAlliedWith(c));
  const m = seatMasks(env.game, a, env.slots);
  assert.deepEqual([b, c, d].map((p) => row(m, Act.BreakAlliance, p)), [1, 1, 0]);
  assert.equal(a.isTraitor(), false);
  act(env, { 0: [Act.BreakAlliance, 0, b.smallID(), 0] });
  assert.ok(a.isTraitor() && !a.isAlliedWith(b));
  assert.equal(b.isTraitor(), false);
  const expires = a.allianceWith(c)!.expiresAt();
  while (env.game.ticks() <= expires) env.tickWith([]);
  assert.ok(!a.isAlliedWith(c) && !c.isAlliedWith(a), "the alliance ran out");
  assert.equal(c.isTraitor(), false);
});

test("donations go to teammates below their cap as a share of the seat's troops; public FFA refuses humans", async () => {
  const hvn = await quickGame(HVN, 5);
  spawnSeats(hvn);
  const [a, b] = hvn.seats;
  const m = seatMasks(hvn.game, a, hvn.slots);
  assert.equal(row(m, Act.DonateTroops, b), 1, "a teammate");
  assert.deepEqual(decodeAction(hvn.game, a, hvn.frame, m, [Act.DonateTroops, 0, b.smallID(), 30]).intent,
    { type: "donate_troops", recipient: b.id(), troops: 0.3 * a.troops() });
  assert.equal(decodeAction(hvn.game, a, hvn.frame, m, [Act.DonateTroops, 0, b.smallID(), 0]).valid, false);
  const [giverBefore, takerBefore] = [a.troops(), b.troops()];
  act(hvn, { 0: [Act.DonateTroops, 0, b.smallID(), 30] });
  assert.ok(a.troops() < 0.8 * giverBefore, "the giver lost about 30%"); // growth over 5 ticks is far smaller
  assert.ok(b.troops() > takerBefore + 0.2 * giverBefore, "the teammate received it");
  const ffa = await quickGame(FFA, 5, LONE);
  spawnSeats(ffa);
  const [c, d] = ffa.seats;
  act(ffa, { 0: [Act.AllianceRequest, 0, d.smallID(), 0], 1: [Act.AllianceRequest, 0, c.smallID(), 0] });
  assert.equal(row(seatMasks(ffa.game, c, ffa.slots), Act.DonateTroops, d), 0, "an allied human, donations off");
});

test("a seat outside the game has no diplomacy and no warship moves", async () => {
  const env = await quickGame(FFA, 5, LONE);
  const zero = (m: Masks) => m.diplomacy.every((v) => v === 0) && DIPLOMACY.every((t) => m.types[t] === 0)
    && m.types[Act.MoveWarships] === 0;
  assert.ok(zero(seatMasks(env.game, env.seats[0], env.slots)), "spawn phase");
  spawnSeats(env, [1, 2, 3]); // seat 0 never spawns
  const m = seatMasks(env.game, env.seats[0], env.slots);
  assert.ok(zero(m), "an unspawned seat after the spawn phase");
  assert.equal(decodeAction(env.game, env.seats[0], env.frame, m,
    [Act.AllianceRequest, 0, env.seats[1].smallID(), 0]).valid, false);
});
