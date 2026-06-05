# Temporal winner-loser networks in WWE, tennis, and football 

Score: 1.2

*Professional wrestling, tennis and football all generate sequences of competitive encounters, but the structure of these encounters is very different. Tennis is mostly one-against-one, football is team-against-team, while WWE/WWF-style wrestling includes singles matches, tag-team matches, and multi-person matches under multiple promotions and historical brands. This project aims to reconstruct temporal winner-loser networks from these three domains and produce comparable network datasets for downstream analysis.*

--

The main dataset should be the WWE/WWF match dataset:

https://www.kaggle.com/datasets/alexdiresta/all-wwe-and-wwf-matches-from-4301979-to-92523

Students should also reconstruct temporal winner-loser networks from tennis and international football data, using for example:

ATP tennis:
https://github.com/JeffSackmann/tennis_atp

WTA tennis:
https://github.com/JeffSackmann/tennis_wta

International football:
https://github.com/martj42/international_results

The main task is to construct temporal directed weighted networks. Nodes are wrestlers, tennis players, or football teams. A directed edge A → B means that A defeated B within a given time window. The weight of the edge is the number of wins of A against B within that window.

For all sports, students should build snapshots of four months. Edge weights should be aggregated within each four-month snapshot. For example, if wrestler A defeats wrestler B three times in the same four-month period, the edge A → B should have weight 3 in that snapshot.

For the WWE/WWF dataset, students should reconstruct a temporal multilayer winner-loser network including WWE, WWF, WWWF, ECW, NXT, and WCW. Promotions should be stored both as original labels and as aggregated organization groups. A recommended aggregation is:

WWE_lineage = WWWF + WWF + WWE

with WCW, ECW, and NXT kept as separate organization groups.

The WWE/WWF network should use the following interaction layers:

1. singles;
2. tag_team;
3. multi_person.

Battle royal, Royal Rumble, and all-vs-all elimination-style matches should be excluded from the mandatory winner-loser network. The reason is that interpreting the winner as having directly defeated every other participant is weaker than in singles, tag-team, or standard multi-person matches, and may artificially create many winner-loser edges from one event. Students may include a battle_royal layer only as an optional robustness analysis, clearly separated from the main network.

WWE/WWF edge-construction rules:

- singles: one winner against one loser gives one edge winner → loser with weight 1;
- tag_team: each member of the winning team is connected to each member of the losing team;
- multi_person: the winner is connected to every non-winning participant;
- for tag-team and multi-person matches, students should document whether they use raw pairwise weights or normalized weights;
- a recommended normalization is 1 / (number of winners × number of losers) for tag-team matches and 1 / number_of_losers for multi-person matches;
- draws, no-contests, unknown outcomes, and ambiguous winners should be excluded from the winner-loser network.

For tennis, students should build directed winner-loser networks from ATP and/or WTA match results. Nodes are players, and an edge player A → player B means that A defeated B in a match. Edge weights are aggregated within four-month snapshots.

For football, students should build a directed winner-loser network of national teams. Nodes are teams, and an edge team A → team B means that A defeated B in an international match. Draws should be excluded from the winner-loser network or stored separately in an optional undirected draw layer.

Expected output:

- WWE/WWF temporal multilayer edge files with columns: `source target weight snapshot_start snapshot_end year promotion organization_group layer`;
- tennis temporal edge files with columns: `source target weight snapshot_start snapshot_end tour`;
- football temporal edge files with columns: `source target weight snapshot_start snapshot_end tournament`;
- node metadata tables for wrestlers, tennis players, and football teams when available;
- scripts to download, clean, parse, and construct the networks;
- a short report documenting data sources, filtering rules, layer definitions, temporal aggregation, and limitations.

The analysis should include:

1. reconstruction of the WWE/WWF temporal multilayer winner-loser network;
2. reconstruction of temporal winner-loser networks for tennis and international football;
3. construction of four-month snapshots for all sports;
4. aggregation of edge weights within each snapshot;
5. separation of WWE/WWF interactions into singles, tag_team, and multi_person layers;
6. exclusion and justification of battle royal and all-vs-all match types from the mandatory network;
7. analysis of WWE/WWF network evolution, including centrality, dominance, communities, and node/edge turnover;
8. identification of highly dominant or central competitors over time;
9. comparison of WWE/WWF organization groups, such as WWE_lineage, WCW, ECW, and NXT;
10. documentation of how ambiguous results, draws, multi-participant events, and missing data were handled.

Optional extensions:

- compare structural properties across WWE/WWF, tennis, and football networks;
- include battle royal matches as a separate robustness layer;
- construct co-participation networks for WWE/WWF, where edges connect wrestlers appearing in the same match;
- compare directed dominance networks with undirected co-participation networks;
- analyze gender-separated tennis networks using ATP and WTA separately;
- compare football networks by tournament type or historical period;
- test whether centrality or dominance rankings are stable across snapshots.