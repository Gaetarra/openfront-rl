import assert from "node:assert/strict";
import path from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";
import {
  Difficulty,
  GameMode,
  HumansVsNations,
  RankedType,
  UnitType,
} from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/game/GameTypes";
import { GameConfig, Intent, StampedIntent } from "../OpenFront_Source_Code/openfrontio/packages/engine-api/src/Schemas";
import { attackMask, BUILDS } from "./actions";
import { EnvGame, seatClientID } from "./game";
import { encodeSeat, KINDS, lobbyKind, MODE, PLAYER_FEATURES, Rel, SCALARS } from "./obs";
import { readRecord, RecordGame } from "./replay";
import { FFA, HVN, LONE, ONION, quickGame, RANKED, script, spawnSeats } from "./testkit";

test("seat view: self, enemies, unowned land row and bounded scalars", async () => {
  const env = await EnvGame.create(ONION, "obs");
  script(env, 260);
  const seat = env.seats[0];
  const P = env.slots;
  const players = new Float32Array(P * PLAYER_FEATURES);
  const rel = new Uint8Array(P);
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.horizonMinutes, env.frame.cs, attack, players, rel, scalars);
  const me = seat.smallID();
  assert.equal(rel[me], Rel.Self);
  assert.deepEqual([...rel].filter((r) => r === Rel.Enemy).length, 3);
  assert.equal(players[me * PLAYER_FEATURES + 2], 1);
  assert.ok(players[7] > 0 && players[7] <= 1, "unowned land share");
  assert.equal(players[12], attack[0]);
  assert.equal(scalars[3], 1, "spawned");
  for (const v of [...players, ...scalars]) assert.ok(Number.isFinite(v) && v >= 0 && v <= 2, `${v}`);
});

test("scalars hold the gold against each build's current cost and the seat's loaded silo missiles", async () => {
  const env = await EnvGame.create(ONION, "costs");
  script(env, 260);
  const seat = env.seats[0];
  seat.addGold(2_000_000n);
  const [tile] = seat.tiles();
  seat.buildUnit(UnitType.MissileSilo, tile, {}).increaseLevel(); // level 2, its new slot reloading: 1 loaded
  seat.buildUnit(UnitType.MissileSilo, tile, {}).setUnderConstruction(true); // none loaded
  const P = env.slots;
  const scalars = new Float32Array(SCALARS);
  const attack = attackMask(env.game, seat, P);
  encodeSeat(env.game, seat, env.horizonMinutes, env.frame.cs, attack, new Float32Array(P * PLAYER_FEATURES),
    new Uint8Array(P), scalars);
  const gold = Number(seat.gold());
  [...BUILDS.values()].forEach((unit, i) => {
    const cost = Number(env.game.unitInfo(unit).cost(env.game, seat));
    assert.equal(scalars[16 + i], Math.fround(Math.min(gold / cost, 4) / 4), unit);
  });
  assert.ok(scalars[16] > scalars[24], "a city costs less than a hydrogen bomb");
  assert.equal(scalars[25], 1 / 16);
});

test("diplomacy features: ally, pending requests, the alliance's time left and who asked to renew", async () => {
  const env = await quickGame(FFA, 5, { ...LONE, customAllianceDuration: 1 });
  spawnSeats(env);
  const g = env.game;
  const [a, b] = env.seats;
  const send = (i: number, intent: Intent) => { env.tickWith([{ ...intent, clientID: seatClientID(i) } as StampedIntent]); env.tickWith([]); };
  const view = () => {
    const players = new Float32Array(env.slots * PLAYER_FEATURES);
    encodeSeat(g, a, env.horizonMinutes, env.frame.cs, attackMask(g, a, env.slots), players, new Uint8Array(env.slots),
      new Float32Array(SCALARS));
    return [17, 18, 19, 20, 21, 22].map((f) => players[b.smallID() * PLAYER_FEATURES + f]);
  };
  send(1, { type: "allianceRequest", recipient: a.id() });
  assert.deepEqual(view(), [0, 1, 0, 0, 0, 0], "b asked a");
  send(0, { type: "allianceRequest", recipient: b.id() });
  const fresh = view();
  assert.deepEqual([fresh[0], fresh[1], fresh[2]], [1, 0, 0], "allied, nothing pending");
  assert.ok(fresh[3] > 0.99, "a new alliance");
  const alliance = a.allianceWith(b)!;
  while (g.ticks() < alliance.expiresAt() - g.config().allianceExtensionPromptOffset()) env.tickWith([]);
  send(1, { type: "allianceExtension", recipient: a.id() });
  const late = view();
  assert.deepEqual([late[4], late[5]], [1, 0], "b asked to renew, a not yet");
  assert.ok(late[3] < 0.5);
});

test("the lobby kind comes from the config's ranked type, mode and teams", () => {
  const kind = (c: Partial<GameConfig>) => lobbyKind({ gameMode: GameMode.FFA, ...c } as GameConfig);
  assert.deepEqual(KINDS, ["ranked1v1", "ranked2v2", "rankedffa", "clanwars", "ffa", "team", "hvn"]);
  assert.deepEqual(
    [
      kind({ rankedType: RankedType.OneVOne }),
      kind({ rankedType: RankedType.TwoVTwo, gameMode: GameMode.Team, playerTeams: 2 }),
      kind({ rankedType: RankedType.FreeForAll }),
      kind({ rankedType: RankedType.ClanWars, gameMode: GameMode.Team }),
      kind({}),
      kind({ gameType: "Private" as GameConfig["gameType"] }),
      kind({ gameMode: GameMode.Team, playerTeams: 6 }),
      kind({ gameMode: GameMode.Team, playerTeams: HumansVsNations }),
    ],
    [0, 1, 2, 3, 4, 4, 5, 6],
  );
});

const modeScalars = (env: EnvGame, i: number) => {
  const scalars = new Float32Array(SCALARS);
  encodeSeat(env.game, env.seats[i], env.horizonMinutes, env.frame.cs, attackMask(env.game, env.seats[i], env.slots),
    new Float32Array(env.slots * PLAYER_FEATURES), new Uint8Array(env.slots), scalars);
  return [...scalars.subarray(MODE, SCALARS)];
};
const kindOneHot = (k: number) => KINDS.map((_, i) => (i === k ? 1 : 0));

test("the mode scalars of ranked 1v1, FFA and Humans vs Nations training games", async () => {
  assert.equal(SCALARS, 36);
  const ranked = await EnvGame.create(RANKED, "mode-ranked");
  for (const i of [0, 1]) {
    assert.deepEqual(modeScalars(ranked, i), [...kindOneHot(0), 1 / 8, 0, 0].map(Math.fround), `ranked seat ${i}`);
  }

  const hvn = await EnvGame.create(HVN, "mode-hvn");
  for (let i = 0; i < hvn.seats.length; i++) {
    assert.deepEqual(modeScalars(hvn, i), [...kindOneHot(6), 0.375, 1 / 6, 0.75].map(Math.fround), `hvn seat ${i}`);
  }

  const ffa = await EnvGame.create(FFA, "mode-ffa");
  const level: Record<string, number> = { [Difficulty.Medium]: 0.5, [Difficulty.Hard]: 0.75 };
  const m = modeScalars(ffa, 0);
  assert.deepEqual(m.slice(0, 7), kindOneHot(4));
  assert.equal(m[8], 0, "no teams");
  assert.equal(m[9], Math.fround(level[ffa.game.config().gameConfig().difficulty]));
  assert.ok(m[7] > 2 / 8, "the seats and the map's nations");
});

test("a record and a training game of the same kind give the same mode scalars", async () => {
  const fixture = path.join(path.dirname(fileURLToPath(import.meta.url)), "testdata/ranked1v1.json.gz");
  const rg = await RecordGame.create(readRecord(fixture), 256, 403);
  const env = await EnvGame.create(RANKED, "mode-same");
  for (let i = 0; i < 2; i++) assert.deepEqual(modeScalars(rg.env, i), modeScalars(env, i), `seat ${i}`);
});
