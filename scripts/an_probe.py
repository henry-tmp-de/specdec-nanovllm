import json, sys, os

for tag in sys.argv[1:]:
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "out", tag + ".json")
    d = json.load(open(p))
    print("=" * 78)
    print(tag, "alpha_step", d["alpha_step"], "rounds", d["n_rounds"],
          "deliv", d["tokens_delivered"], "W", d["window"], "P", d["head"], "B", d["B"], "ctx", d["ctx"])
    hdr = ("{:>3} {:>6} {:>3} {:>5} {:>5} {:>8} {:>8} {:>10} {:>8} {:>4} {:>6}".format(
        "i", "L", "b0", "clen", "nwin", "tvAll", "tv_j0", "eng~win", "maxabs", "acc", "p~win"))
    print(hdr)
    for r in d["rounds"]:
        pr = r.get("probe")
        if not pr:
            continue
        print("{:>3} {:>6} {:>3} {:>5} {:>5} {:>8.4f} {:>8.4f} {:>10.5f} {:>8.3f} {:>4} {:>6}".format(
            r["i"], pr["L"], pr["b0"], pr["ctx_len"], pr["nwin"],
            r.get("tv_all", -1), r["tv_by_j"][0], pr["tv_eng_vs_win"],
            pr["maxabs_eng_vs_win"], r["acc"], pr.get("tv_p_eng_vs_win", -1)))
