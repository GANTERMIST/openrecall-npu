"""Гибридный поиск: смысловое сходство векторов + бонус за точные слова (SQLite FTS5).
Общий для Recall и поиска по файлам."""
import sqlite3

import numpy as np


def hybrid(con, query_vec, query: str, emb_sql: str, fts_table: str, k: int = 10, bonus=0.15):
    """emb_sql возвращает строки (id, vec_blob). Возвращает [(id, score)] по убыванию."""
    scores = {}
    rows = con.execute(emb_sql).fetchall()
    if rows:
        ids = np.array([r[0] for r in rows])
        mat = np.stack([np.frombuffer(r[1], np.float32) for r in rows])
        sims = mat @ query_vec.astype(np.float32)
        for i, s in zip(ids, sims):
            i = int(i)
            scores[i] = max(scores.get(i, -1.0), float(s))
    words = [w for w in query.replace('"', " ").split() if len(w) > 2]
    if words:
        try:
            fq = " OR ".join(f'"{w}"*' for w in words)
            for rid, in con.execute(f"SELECT rowid FROM {fts_table} WHERE {fts_table} MATCH ? LIMIT 200", (fq,)):
                scores[rid] = scores.get(rid, 0.0) + bonus
        except sqlite3.OperationalError:
            pass
    return sorted(scores.items(), key=lambda x: -x[1])[:k]
