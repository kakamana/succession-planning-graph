"use client";

import { useEffect, useState } from "react";

const API = process.env.NEXT_PUBLIC_API ?? "http://localhost:8000";

type Candidate = {
  emp_id: string;
  role: string;
  level: number;
  tenure_yrs: number;
  skill_match: number;
  structural_proximity: number;
  performance: number;
  readiness_score: number;
  shortest_path_distance: number;
};

type Resp = {
  manager_id: string;
  candidates: Candidate[];
  decision_aid_disclaimer: string;
};

const fmt = (x: number) => (x * 100).toFixed(1) + "%";

export default function Home() {
  const [managers, setManagers] = useState<string[]>([]);
  const [managerId, setManagerId] = useState("E-0001");
  const [k, setK] = useState(5);
  const [data, setData] = useState<Resp | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetch(`${API}/managers`).then((r) => r.json()).then((m: string[]) => {
      setManagers(m);
      if (m.length > 0) setManagerId(m[0]);
    });
  }, []);

  async function run() {
    setLoading(true);
    const res = await fetch(`${API}/succession`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ manager_id: managerId, k }),
    });
    setData(await res.json());
    setLoading(false);
  }

  return (
    <main className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold">Succession-Planning Graph</h1>
      <p className="opacity-70 mb-6">
        Pick a manager. Get the top-k succession candidates with three sub-scores and a blended readiness.
      </p>

      <div className="flex flex-wrap items-end gap-3">
        <label className="flex flex-col">
          <span className="text-xs opacity-60 mb-1">Manager ID</span>
          <input
            list="managers"
            className="border rounded-xl p-2 min-w-[12rem]"
            value={managerId}
            onChange={(e) => setManagerId(e.target.value)}
          />
          <datalist id="managers">
            {managers.map((m) => (
              <option key={m} value={m} />
            ))}
          </datalist>
        </label>
        <label className="flex flex-col">
          <span className="text-xs opacity-60 mb-1">k (candidates)</span>
          <input
            type="number"
            min={1}
            max={25}
            className="border rounded-xl p-2 w-24"
            value={k}
            onChange={(e) => setK(Number(e.target.value))}
          />
        </label>
        <button
          onClick={run}
          disabled={loading}
          className="rounded-xl px-4 py-2 bg-black text-white disabled:opacity-50"
        >
          {loading ? "Ranking..." : "Find successors"}
        </button>
      </div>

      {data && (
        <>
          <div className="mt-8 grid gap-4">
            {data.candidates.map((c) => (
              <CandidateCard key={c.emp_id} c={c} />
            ))}
          </div>
          <p className="mt-6 text-xs opacity-60 italic">
            {data.decision_aid_disclaimer}
          </p>
        </>
      )}
    </main>
  );
}

function CandidateCard({ c }: { c: Candidate }) {
  return (
    <div className="rounded-2xl border p-4">
      <div className="flex justify-between items-start">
        <div>
          <div className="text-lg font-semibold">{c.emp_id}</div>
          <div className="text-xs opacity-70">
            {c.role} · L{c.level} · {c.tenure_yrs.toFixed(1)} yrs · spd {c.shortest_path_distance}
          </div>
        </div>
        <div className="text-right">
          <div className="text-xs uppercase opacity-60">Readiness</div>
          <div className="text-2xl font-semibold">{fmt(c.readiness_score)}</div>
        </div>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3">
        <SubScore label="Skill match" value={c.skill_match} color="bg-emerald-500" />
        <SubScore label="Structural" value={c.structural_proximity} color="bg-sky-500" />
        <SubScore label="Performance" value={c.performance} color="bg-amber-500" />
      </div>
    </div>
  );
}

function SubScore({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div>
      <div className="text-xs opacity-60">{label}</div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden mt-1">
        <div
          className={`h-full ${color}`}
          style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }}
        />
      </div>
      <div className="text-xs mt-1">{fmt(value)}</div>
    </div>
  );
}
