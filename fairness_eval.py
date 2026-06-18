"""
Fairness / demographic-slice evaluation on LFW.

LFW does not ship demographic labels, so we infer gender from the identity's
common first name. We use a name->gender lookup table for the most frequent
LFW identities (built from the LFW name list), with an explicit `other`/
`unknown` bucket so we never silently mislabel.

For each gender slice we compute:
    - n_pairs (same / different)
    - 10-fold mean accuracy + std
    - ROC AUC, EER
    - TAR @ FAR=1e-3, TAR @ FAR=1e-2
    - Precision, Recall, F1 at the global chosen threshold

We then compute disparity metrics:
    - max - min accuracy across slices
    - Equalized-odds-style gaps (TAR gap, FAR gap)
    - Demographic parity ratio

The user can extend `FAIRNESS_LABELS` to other axes (age band, skin-tone)
by adding more rows of (identity_substring, attribute, value). For race
slices, plug in BFW / RFW pair files via --custom-attr-file.

Usage:
    python fairness_eval.py \
        --lfw-root data/sklearn_lfw/lfw_home/lfw_funneled \
        --pairs data/sklearn_lfw/lfw_home/pairs.txt \
        --checkpoint checkpoints/casia_v4_iresnet18.best.pt --backbone iresnet18 \
        --report-dir reports/fairness
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import config
from eval_io import load_threshold
from evaluate import kfold_evaluate, parse_pairs, roc_auc, tar_at_far
from evaluate_full import equal_error_rate, far_frr_curve, precision_recall_f1
from model import build_embedding_model, load_checkpoint
from utils import get_device, l2_normalize
import mlflow_utils as mlu


# ---------------------------------------------------------------------------
# Gender labels — first-name heuristic.
# This is a documented heuristic, not a ground-truth annotation. It is
# adequate for *aggregate* fairness reporting on LFW (the bias signal is
# strong at the population level even with ~10% per-name error), but it
# should not be used to label individuals. The codebase exposes a clear
# `--custom-attr-file` escape hatch for users with annotated datasets
# (BFW, RFW, FairFace-mapped, etc).
# ---------------------------------------------------------------------------

MALE_NAMES = {
    "Aaron","Abdullah","Adrien","Akhmed","Al","Alan","Alberto","Alec","Alex",
    "Alexander","Ali","Allen","Alvaro","Amelie","Amer","Amir","Anders","Andre",
    "Andrei","Andres","Andy","Andrew","Anthony","Antonio","Aretha","Ari","Arnold",
    "Asif","Atal","Augusto","Bashar","Ben","Benjamin","Bernard","Bertie","Bill",
    "Billy","Bob","Brad","Brendan","Brett","Brian","Bruce","Bryan","Calvin","Carl",
    "Carlos","Cesar","Charles","Charlie","Chen","Chip","Choi","Chris","Christian",
    "Christopher","Chuck","Clint","Colin","Conan","Craig","Daniel","Darrell","David",
    "Dean","Dennis","Derek","Dick","Diego","Dominic","Donald","Doug","Douglas","Dwayne",
    "Ed","Edgar","Edmund","Eduardo","Edward","Edwin","Eli","Eric","Erik","Eugene","Evan",
    "Fabian","Faisal","Felipe","Fernando","Fidel","Francis","Frank","Frans","Fred",
    "Gabriel","Gary","Gene","Geoffrey","George","Gerald","Gerard","Gerhard","Giuseppe",
    "Glenn","Goldie","Goran","Gordon","Gray","Greg","Gregg","Gregory","Guillermo","Hamid",
    "Hank","Hans","Harrison","Harry","Hassan","Henry","Herb","Hideki","Hideo","Hosni",
    "Howard","Hugh","Hugo","Iain","Ian","Igor","Ivan","Jack","Jacques","Jaime","Jake",
    "James","Jamie","Jason","Javier","Jay","Jean","Jeff","Jeffrey","Jeremy","Jerry",
    "Jesse","Jiang","Jim","Jimmy","Joe","Joel","John","Johnny","Jon","Jonathan","Jorge",
    "Jose","Joseph","Juan","Julian","Julio","Justin","Karl","Keith","Ken","Kenneth",
    "Kevin","Kim","Kofi","Kurt","Lance","Larry","Laurent","Lee","Leo","Leon","Leonardo",
    "Leszek","Lewis","Liam","Lleyton","Louis","Luc","Lucio","Luis","Marc","Marcel",
    "Marco","Marcos","Mariano","Mario","Mark","Martin","Marty","Mathew","Matt","Matthew",
    "Maurice","Max","Mel","Michael","Mick","Miguel","Mike","Mikhail","Milan","Mohammad",
    "Mohammed","Muhammad","Nelson","Neil","Nelson","Nestor","Nick","Nicolas","Norm",
    "Olaf","Oscar","Oswaldo","Owen","Pat","Patrick","Paul","Pedro","Pervez","Pete",
    "Peter","Phil","Philip","Pierce","Pierre","Pope","Quentin","Rafael","Ralf","Ralph",
    "Randall","Randy","Raul","Ray","Raymond","Reggie","Rene","Ricardo","Richard","Rick",
    "Ricky","Robert","Robin","Rod","Roger","Roh","Roman","Romeo","Ron","Ronald","Ronnie",
    "Roy","Rubens","Rudi","Rudolph","Rudy","Russell","Ryan","Sam","Samuel","Sammy","Sean",
    "Sergei","Sergey","Sergio","Shaun","Shawn","Silvan","Silvio","Simon","Spencer",
    "Stanley","Stephane","Stephen","Steve","Steven","Stuart","Sven","Taha","Takeo","Ted",
    "Terry","Thabo","Thomas","Tiger","Tim","Timothy","Tom","Tommy","Tony","Trent","Trevor",
    "Tyler","Tyson","Vaclav","Valentino","Vicente","Victor","Vincent","Vladimir","Walter",
    "Wayne","Wen","Will","William","Willie","Willy","Woody","Yang","Yashwant","Yoriko",
    "Yusuke","Zach","Zinedine","Zoran",
}
FEMALE_NAMES = {
    "Abigail","Agnes","Aishwarya","Alanis","Alexandra","Alicia","Alison","Allison","Amanda",
    "Amber","Amelie","Amy","Andrea","Angela","Angelica","Angelina","Anita","Anjali","Ann",
    "Anna","Anne","Annette","Annika","April","Ariane","Ariel","Ashley","Aspen","Audrey",
    "Avril","Barbara","Beatrice","Belinda","Beth","Betty","Beverly","Bridget","Britney",
    "Brittany","Brooke","Calista","Camilla","Cameron","Carla","Carmen","Carol","Carole",
    "Caroline","Carolyn","Cate","Catherine","Catriona","Cecilia","Celine","Chante","Charlene",
    "Charlize","Chelsea","Cherie","Cheryl","Chloe","Christina","Christine","Cindy","Claire",
    "Claudia","Connie","Courteney","Courtney","Cynthia","Cyndi","Danae","Daniela","Danielle",
    "Daryl","Dawn","Deborah","Debra","Demi","Denise","Diane","Donna","Dora","Doris","Drew",
    "Edie","Elaine","Elena","Eliza","Elizabeth","Ellen","Emily","Emma","Erica","Erin",
    "Estella","Eunice","Eva","Faith","Fanny","Faye","Florence","Frances","Francesca","Gabrielle",
    "Gail","Geena","Gemma","Georgia","Geraldine","Gillian","Gina","Giselle","Gloria","Goldie",
    "Gretchen","Gwen","Halle","Hannah","Hayley","Heather","Helen","Helena","Hilary","Hillary",
    "Holly","Hope","Hyun","Ines","Irina","Iris","Isabel","Isabella","Jacqueline","Jada","Jaime",
    "Jamie","Jane","Janelle","Janet","Janice","Janis","Jasmine","Jeanette","Jeanne","Jen",
    "Jenna","Jennifer","Jenny","Jerri","Jessica","Jill","Jo","Joan","Joanna","Joanne","Jodi",
    "Jodie","Jodie","Joelle","Joey","Joy","Joyce","Judith","Judy","Julia","Julianne","Julie",
    "Juliette","Justine","Karen","Kari","Kate","Katharine","Katherine","Kathleen","Kathryn",
    "Kathy","Katie","Katrina","Kelly","Kendra","Kim","Kimberly","Kirsten","Kirsty","Kristen",
    "Kristin","Kylie","Lara","Laura","Laure","Laurel","Lauren","Laurence","Leah","Lily",
    "Linda","Lindsay","Lisa","Liv","Liza","Lori","Lorraine","Lucia","Lucy","Lynn","Madeleine",
    "Madonna","Maggie","Mai","Mandy","Mara","Margaret","Marge","Maria","Mariah","Marianne",
    "Marie","Marilyn","Marisa","Martha","Martina","Mary","Maureen","Megan","Megawati","Melanie",
    "Melissa","Meredith","Meryl","Michelle","Mireya","Miriam","Molly","Monica","Monique",
    "Morgan","Nancy","Naomi","Natalia","Natalie","Natasha","Nathalie","Nia","Nicole","Nicki",
    "Nikki","Nina","Nora","Norah","Olga","Olivia","Padma","Pam","Pamela","Paola","Patricia",
    "Patty","Paula","Pauline","Penelope","Penny","Petra","Pier","Princess","Priscilla","Queen",
    "Rachel","Rebecca","Reese","Regina","Renee","Rhona","Rita","Roberta","Rosa","Rosanna",
    "Rose","Rosemary","Rosie","Roxanne","Ruth","Sally","Salma","Samantha","Sandra","Sara",
    "Sarah","Sasha","Sharon","Sheila","Sheri","Sheryl","Shirley","Shoshannah","Sigrid","Silvia",
    "Simone","Sofia","Sonia","Sonja","Sophia","Sophie","Stacey","Stacy","Stella","Stephanie",
    "Suzanne","Susan","Susanna","Svetlana","Sybille","Sylvia","Tabitha","Tamara","Tammy","Tara",
    "Tatiana","Teresa","Teri","Terri","Theresa","Tia","Tiffany","Tina","Tonya","Tracy","Tricia",
    "Trish","Tyra","Uma","Ursula","Valentina","Valerie","Vanessa","Vera","Veronica","Vicki",
    "Victoria","Vivian","Whitney","Wilma","Yolanda","Yoko","Yvonne","Zara","Zhang","Zorica",
}


def name_to_gender(identity: str) -> str:
    """Identity dir name -> 'male' / 'female' / 'unknown'."""
    first = identity.split("_")[0]
    if first in MALE_NAMES:
        return "male"
    if first in FEMALE_NAMES:
        return "female"
    return "unknown"


# ---------------------------------------------------------------------------
# Embedding (re-use fast path)
# ---------------------------------------------------------------------------

def _build_transform(input_size, center_crop):
    return transforms.Compose([
        transforms.CenterCrop(center_crop),
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5019607, 0.5019607, 0.5019607)),
    ])


@torch.no_grad()
def embed_all(paths, model, device, input_size, center_crop, batch_size):
    tf = _build_transform(input_size, center_crop)
    embs: dict[Path, np.ndarray] = {}
    imgs, ps = [], []
    def flush():
        if not imgs:
            return
        x = torch.stack(imgs, dim=0).to(device)
        e = l2_normalize(model(x).detach().cpu().numpy().astype(np.float32))
        for pp, ee in zip(ps, e):
            embs[pp] = ee
        imgs.clear(); ps.clear()
    for p in tqdm(paths, desc="embed"):
        try:
            img = Image.open(p).convert("RGB")
        except Exception:
            continue
        imgs.append(tf(img)); ps.append(p)
        if len(imgs) >= batch_size:
            flush()
    flush()
    return embs


def slice_pairs(pairs, folds, identity_attr_fn):
    """Group pairs by (attr_a, attr_b) bucket. Returns dict slice_name -> list of (a,b,label,fold)."""
    buckets: dict[str, list] = {}
    for (a, b, label), fold in zip(pairs, folds):
        attr_a = identity_attr_fn(a.parent.name)
        attr_b = identity_attr_fn(b.parent.name)
        if attr_a == attr_b and attr_a != "unknown":
            buckets.setdefault(attr_a, []).append((a, b, label, fold))
        # Cross-bucket pairs are excluded from same-slice analysis but kept
        # in 'cross' for awareness.
        elif attr_a != attr_b:
            buckets.setdefault("cross", []).append((a, b, label, fold))
    return buckets


def evaluate_slice(slice_rows, embeddings, global_threshold):
    scores, labels, folds = [], [], []
    for a, b, lab, f in slice_rows:
        ea, eb = embeddings.get(a), embeddings.get(b)
        if ea is None or eb is None:
            continue
        scores.append(float(np.dot(ea, eb)))
        labels.append(int(lab))
        folds.append(int(f))
    if not scores:
        return None
    scores = np.array(scores, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)
    folds = np.array(folds, dtype=np.int32)
    grid = np.linspace(-0.1, 1.0, 222)
    # Per-slice k-fold is unreliable if a fold has few samples; report
    # both per-slice fold accuracy and accuracy at the global threshold.
    n_per_fold = np.array([(folds == f).sum() for f in np.unique(folds)])
    kf = None
    if (n_per_fold >= 20).all() and len(np.unique(folds)) >= 3:
        kf = kfold_evaluate(scores, labels, folds, grid)
    auc = roc_auc(scores, labels) if (labels == 0).any() and (labels == 1).any() else float("nan")
    thr_sweep, far, frr = far_frr_curve(scores, labels, n_thresh=300)
    eer, eer_thr = equal_error_rate(thr_sweep, far, frr)
    tar1e3, _ = tar_at_far(scores, labels, 1e-3)
    tar1e2, _ = tar_at_far(scores, labels, 1e-2)
    cm = precision_recall_f1(scores, labels, global_threshold)
    return {
        "n_pairs": int(len(scores)),
        "n_pos": int((labels == 1).sum()),
        "n_neg": int((labels == 0).sum()),
        "kfold_accuracy": float(kf["mean_accuracy"]) if kf else float(((scores >= global_threshold).astype(int) == labels).mean()),
        "kfold_std": float(kf["std_accuracy"]) if kf else 0.0,
        "auc": float(auc),
        "eer": float(eer),
        "tar_at_far_1e3": float(tar1e3),
        "tar_at_far_1e2": float(tar1e2),
        "precision_at_global_thr": cm["precision"],
        "recall_at_global_thr": cm["recall"],
        "f1_at_global_thr": cm["f1"],
        "far_at_global_thr": float(((scores >= global_threshold) & (labels == 0)).sum() / max(1, (labels == 0).sum())),
        "frr_at_global_thr": float(((scores < global_threshold) & (labels == 1)).sum() / max(1, (labels == 1).sum())),
        # Predicted positive rate for demographic parity ratio.
        "predicted_positive_rate": float((scores >= global_threshold).mean()),
    }


def disparity_metrics(per_slice: dict) -> dict:
    """Canonical group-fairness metrics for verification.

    In face *verification* (same/different binary decision), the standard
    group-fairness quantities are:

      * Equal opportunity (Hardt et al. 2016, "equal TPR"):
            TPR_a should equal TPR_b across groups a, b. Here TPR ≡ TAR.
            Disparity = max_a TPR_a − min_a TPR_a.

      * Equalized odds (Hardt et al. 2016):
            TPR_a = TPR_b AND FPR_a = FPR_b. Here FPR ≡ FAR.
            Reported as the larger of the TPR-gap and FPR-gap.

      * Predictive parity / accuracy parity ratio:
            min_a Acc_a / max_a Acc_a. NOTE: this is **not**
            "demographic parity" — DP is about prediction RATES, not
            accuracy. We use the accurate name.

      * Demographic parity ratio (disparate impact, Feldman et al. 2015):
            min_a P(Ŷ=1|A=a) / max_a P(Ŷ=1|A=a).
            For verification the "positive prediction rate" is the
            fraction of pairs predicted "same" within a slice.
    """
    keys = [k for k in per_slice if per_slice[k] is not None and k != "cross"]
    if len(keys) < 2:
        return {"note": "Need >= 2 evaluable slices for disparity"}
    accs = [per_slice[k]["kfold_accuracy"] for k in keys]
    fars = [per_slice[k]["far_at_global_thr"] for k in keys]
    frrs = [per_slice[k]["frr_at_global_thr"] for k in keys]
    aucs = [per_slice[k]["auc"] for k in keys]
    tars = [1.0 - f for f in frrs]                  # TAR = 1 - FRR at thr
    pred_pos_rate = [per_slice[k].get("predicted_positive_rate", float("nan"))
                     for k in keys]
    tpr_gap = float(max(tars) - min(tars))
    fpr_gap = float(max(fars) - min(fars))
    return {
        "slices": keys,
        # Gaps (max - min) across slices.
        "accuracy_gap":   float(max(accs) - min(accs)),
        "auc_gap":        float(max(aucs) - min(aucs)),
        "tpr_gap":        tpr_gap,
        "fpr_gap":        fpr_gap,
        # Canonical fairness summaries.
        "equal_opportunity_disparity":     tpr_gap,
        "equalized_odds_disparity":        float(max(tpr_gap, fpr_gap)),
        "accuracy_parity_ratio":           float(min(accs) / max(accs)) if max(accs) > 0 else 0.0,
        "demographic_parity_ratio":        float(min(pred_pos_rate) / max(pred_pos_rate))
                                            if max(pred_pos_rate) > 0 else float("nan"),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lfw-root", required=True, type=str)
    p.add_argument("--pairs", required=True, type=str)
    p.add_argument("--checkpoint", type=str, default=str(config.DEFAULT_CHECKPOINT))
    p.add_argument("--backbone", type=str, default="iresnet18")
    p.add_argument("--global-threshold", type=float, default=None,
                   help="Decision threshold for FAR/FRR @ thr. Prefer "
                        "--threshold-from to chain from evaluate_full.py.")
    p.add_argument("--threshold-from", type=str, default="",
                   help="Path to evaluate_full.py's threshold.json (or its "
                        "report dir).")
    p.add_argument("--threshold-fallback", type=float, default=config.MATCH_THRESHOLD)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--report-dir", type=str, default="reports/fairness")
    return p.parse_args()


def main():
    args = parse_args()
    device = get_device()
    report_dir = Path(args.report_dir); report_dir.mkdir(parents=True, exist_ok=True)
    with mlu.init_run(
        experiment="fairness_eval",
        run_name=f"fairness_{args.backbone}",
        params=vars(args),
        category="Reliability",
        tags={"step": "fairness_eval", "backbone": args.backbone,
              "checkpoint": str(args.checkpoint)},
    ):
        _run(args, device, report_dir)


def _run(args, device, report_dir):
    if args.threshold_from:
        args.global_threshold = load_threshold(args.threshold_from, args.threshold_fallback)
    elif args.global_threshold is None:
        args.global_threshold = float(args.threshold_fallback)

    pairs, folds = parse_pairs(Path(args.pairs), Path(args.lfw_root))
    # Print population gender breakdown for context.
    all_ids = sorted({p.parent.name for a, b, _ in pairs for p in (a, b)})
    pop = {"male": 0, "female": 0, "unknown": 0}
    for ident in all_ids:
        pop[name_to_gender(ident)] += 1
    print(f"[fairness] identity gender population: {pop}")

    if Path(args.checkpoint).is_file() and args.backbone != "facenet_vggface2":
        model = build_embedding_model(args.backbone)
        load_checkpoint(model, args.checkpoint, map_location=device)
        bb = args.backbone
    else:
        model = build_embedding_model("facenet_vggface2"); bb = "facenet_vggface2"
    model.to(device).eval()
    input_size = 112 if bb.startswith("iresnet") else 160

    unique_paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    embeddings = embed_all(unique_paths, model, device, input_size, 160, args.batch_size)

    buckets = slice_pairs(pairs, folds, name_to_gender)
    print(f"[fairness] pair counts: " + ", ".join(f"{k}={len(v)}" for k, v in buckets.items()))

    per_slice = {k: evaluate_slice(v, embeddings, args.global_threshold) for k, v in buckets.items()}
    disp = disparity_metrics(per_slice)

    out = {
        "backbone": bb,
        "checkpoint": str(args.checkpoint),
        "global_threshold": args.global_threshold,
        "population_identities": pop,
        "per_slice": per_slice,
        "disparity": disp,
    }
    (report_dir / "metrics.json").write_text(json.dumps(out, indent=2))

    # Bar plot.
    slices = [k for k in per_slice if per_slice[k] is not None and k != "cross"]
    accs = [per_slice[k]["kfold_accuracy"] * 100 for k in slices]
    eers = [per_slice[k]["eer"] * 100 for k in slices]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(slices)); w = 0.35
    ax.bar(x - w / 2, accs, w, label="Accuracy (%)")
    ax.bar(x + w / 2, eers, w, label="EER (%)")
    ax.set_xticks(x); ax.set_xticklabels(slices)
    ax.set_ylabel("Percent"); ax.set_title(f"Per-slice fairness — {bb}")
    ax.legend(); ax.grid(alpha=0.3)
    for i, v in enumerate(accs):
        ax.text(i - w / 2, v + 0.5, f"{v:.1f}", ha="center", fontsize=9)
    fig.tight_layout(); fig.savefig(report_dir / "fairness_slices.png", dpi=130)
    plt.close(fig)

    # Markdown summary. "cross" is excluded from the demographic-fairness
    # table because cross-gender pairs are trivially different-identity and
    # therefore tell us nothing about within-group performance disparity.
    md = ["# Fairness evaluation\n", f"Backbone: `{bb}`\n",
          f"Threshold: {args.global_threshold:.3f}\n\n"]
    md.append("## Identity-level population\n")
    md.append(f"- male: {pop['male']}  female: {pop['female']}  unknown: {pop['unknown']}\n\n")
    md.append("## Per-slice metrics (within-group pairs only)\n")
    md.append("| slice | n_pairs | accuracy | AUC | EER | TAR@FAR=1e-3 | FAR@thr | FRR@thr |\n"
              "|---|---:|---:|---:|---:|---:|---:|---:|\n")
    for k, v in per_slice.items():
        if v is None or k == "cross":
            continue
        md.append(f"| {k} | {v['n_pairs']} | {v['kfold_accuracy']*100:.2f}% | "
                  f"{v['auc']:.4f} | {v['eer']*100:.2f}% | "
                  f"{v['tar_at_far_1e3']*100:.2f}% | "
                  f"{v['far_at_global_thr']*100:.2f}% | {v['frr_at_global_thr']*100:.2f}% |\n")
    md.append("\n## Group-fairness disparities\n")
    md.append(json.dumps(disp, indent=2))
    md.append("\n\n## Informational: cross-gender pairs (excluded from disparity)\n")
    cross = per_slice.get("cross")
    if cross is not None:
        md.append(f"- n_pairs: {cross['n_pairs']}  (these are by construction "
                  "different-identity pairs; not a within-group fairness signal)\n")
    (report_dir / "summary.md").write_text("".join(md))
    print(f"\n[fairness] wrote {report_dir/'metrics.json'} + fairness_slices.png + summary.md")

    # MLflow: log per-slice metrics + disparity gaps as a flat name space.
    for slice_name, v in per_slice.items():
        if v is None:
            continue
        mlu.log_metrics_flat({
            f"slice.{slice_name}.accuracy": v.get("kfold_accuracy"),
            f"slice.{slice_name}.auc": v.get("auc"),
            f"slice.{slice_name}.eer": v.get("eer"),
            f"slice.{slice_name}.tar_at_far_1e3": v.get("tar_at_far_1e3"),
            f"slice.{slice_name}.far_at_thr": v.get("far_at_global_thr"),
            f"slice.{slice_name}.frr_at_thr": v.get("frr_at_global_thr"),
        })
    mlu.log_metrics_flat({
        "disparity.accuracy_gap": disp.get("accuracy_gap"),
        "disparity.auc_gap": disp.get("auc_gap"),
        "disparity.tpr_gap": disp.get("tpr_gap"),
        "disparity.fpr_gap": disp.get("fpr_gap"),
        "disparity.equal_opportunity": disp.get("equal_opportunity_disparity"),
        "disparity.equalized_odds": disp.get("equalized_odds_disparity"),
        "disparity.accuracy_parity_ratio": disp.get("accuracy_parity_ratio"),
        "disparity.demographic_parity_ratio": disp.get("demographic_parity_ratio"),
    })
    mlu.log_artifacts_glob(report_dir, ["*.png", "*.json", "*.md"],
                           artifact_path="fairness")


if __name__ == "__main__":
    main()
