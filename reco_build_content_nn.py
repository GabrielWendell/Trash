# -*- coding: utf-8 -*-
df.sort_values("mtime", ascending=False, inplace=True)
df = df.drop_duplicates(subset=["id"]).reset_index(drop=True)


# vectors + sims
Z = build_vectors(df)
nbrs, sims = topk_neighbors(Z, k=args.k)


# popularity + recency
logs = read_logs(Path(args.logs_csv)) if args.logs_csv else pd.DataFrame()
pop, rec = popularity_recency(df, logs, tau_days=args.tau_days)


# store score and diversification
# relative relevance for MMR: normalize pop×(0.2+0.8*rec) or 0.8*pop+0.2*rec
rel = {}
for aid in df["id"]:
rel[aid] = 0.8 * pop.get(aid, 0.0) + 0.2 * rec.get(aid, 0.0)
# order by rel and diversify
order = sorted(df["id"].tolist(), key=lambda x: rel.get(x, 0.0), reverse=True)
id_to_idx = {aid: i for i, aid in enumerate(df["id"]) }
store_ids = mmr_rank(order, rel, sims, id_to_idx, k=args.store_k, lam=0.7)


# write artifacts locally
out = Path(args.out) / "index"
(out / "cont_neighbors").mkdir(parents=True, exist_ok=True)
(out / "meta").mkdir(parents=True, exist_ok=True)


id_list = df["id"].tolist()
for i, aid in enumerate(id_list):
nn_ids = [id_list[j] for j in nbrs[i]]
(out / "cont_neighbors" / f"{aid}.json").write_text(json.dumps(nn_ids, ensure_ascii=False, indent=2), encoding="utf8")


(out / "store_top.json").write_text(json.dumps(store_ids, ensure_ascii=False, indent=2), encoding="utf8")


meta = {aid: {"name": n, "owner": o, "scope": s} for aid, n, o, s in zip(df["id"], df["name"], df["owner"], df["scope"]) }
(out / "meta" / "agents_min.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf8")


print(f"Artifacts written under: {out.resolve()}")


# optional S3 push
if args.push_s3:
bucket = os.getenv("LOG_BUCKET")
if not bucket:
raise SystemExit("LOG_BUCKET env var not set")
prefix = "agents/index/"
# push store
s3_put(bucket, prefix + "store_top.json", (out / "store_top.json").read_bytes())
s3_put(bucket, prefix + "meta/agents_min.json", (out / "meta" / "agents_min.json").read_bytes())
# neighbors
for f in (out / "cont_neighbors").glob("*.json"):
s3_put(bucket, prefix + f"cont_neighbors/{f.name}", f.read_bytes())
print(f"Pushed to s3://{bucket}/{prefix}")


if __name__ == "__main__":
if len(sys.argv) == 1:
print("\nHint: pass --root-base 'data/Agents Chatbots/s3_agents_download' (quotes needed if path has spaces).\n")
main()