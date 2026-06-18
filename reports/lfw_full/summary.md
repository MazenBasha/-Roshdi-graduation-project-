# Full LFW evaluation
## audit_iresnet18
- Backbone: `iresnet18`
- Pairs scored: 6000
- 10-fold accuracy: **79.65% ± 1.23%**  (95% CI 78.95–80.72)
- ROC AUC: **0.8809**  (95% CI 0.8747–0.8886)
- EER: **20.29%** @ threshold 0.354
- P/R/F1 @ chosen threshold 0.375: 0.821 / 0.760 / 0.790
- Operating points:
  - FAR=0.1: thr=0.413  TAR=68.57%  FRR=31.43%
  - FAR=0.01: thr=0.543  TAR=39.53%  FRR=60.47%
  - FAR=0.001: thr=0.641  TAR=19.50%  FRR=80.50%
  - FAR=0.0001: thr=0.675  TAR=14.27%  FRR=85.73%

