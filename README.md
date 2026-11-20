# Intelligent EV Charging and Discharging Scheduling System with Wired and Wireless Infrastructures

**Authors:** Yao Tang, Kwok Tong Chau, Wei Liu, Yunhe Hou  
**Code release date:** 19 November 2026

This repository contains the Proposed-DDPG method and its ablations. See the paper for details.

## Run

`main.py` is the entry point:

```bash
python main.py --evs 10
python main.py --evs 40
```

Results are stored in `results/10ev` or `results/40ev`. Matching completed runs are skipped, and their results table is displayed.



Following shows the training from scratch
<img width="949" height="421" alt="image" src="https://github.com/user-attachments/assets/c2bc3f6b-2de4-45cb-bdc1-780048983c5a" />
<img width="948" height="431" alt="image" src="https://github.com/user-attachments/assets/cdfd5c1d-cc2e-4b1b-af75-95e33dbfd11a" />
