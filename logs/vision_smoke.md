# Vision smoke (Gemini vs SigLIP)

- Sample n=500 (NR=350)
- Download probe (first 50): 44/50 (88%)

## Gemini (`gemini-3.6-flash`, medium res, thinking=MINIMAL)

- ok/coverage: 459 / 91.8%
- est_cost_usd (this run tokens): 0.831
- **cv ridge spearman(rate): 0.0279**
- in-sample ridge spearman(rate): 0.1097

| feature | spearman(rate) | spearman(log_eng) | mean |
|---|---:|---:|---:|
| has_text_in_image | -0.0190 | -0.0203 | 0.725 |
| is_screenshot | -0.0217 | -0.0232 | 0.407 |
| is_meme | 0.0388 | 0.0388 | 0.172 |
| is_chart | -0.0331 | -0.0334 | 0.068 |
| has_face | 0.0142 | 0.0163 | 0.368 |
| is_photo_realworld | 0.0902 | 0.0919 | 0.407 |
| humor_0_10 | 0.0648 | 0.0660 | 1.782 |
| controversy_0_10 | 0.0930 | 0.0948 | 0.932 |

## SigLIP (`google/siglip-base-patch16-224`)

- ok/coverage: 459 / 91.8%
- dims: 768
- pc1 spearman(rate): -0.0335
- best dim spearman(rate): 0.1539
- **cv ridge spearman(rate): 0.0674**
- cv PCA-8 ridge spearman(rate): 0.1006

## Read

- Primary smoke metric is **5-fold CV ridge Spearman vs exposure rate** (not val `wa_*`).
- In-sample probes with dim ≫ n are ignored (overfit).
- Prefer SigLIP if its CV probe ≥ Gemini CV combo; pay for Gemini only if OCR/structure tags clearly add complementary signal.
- Full-corpus spend (~10k media) only after CV corr clears ~0.10+ with decent coverage.

```json
{
  "n_sample": 500,
  "n_nonreply": 350,
  "download_probe": {
    "n": 50,
    "ok": 44,
    "rate": 0.88
  },
  "gemini": {
    "n_ok": 459,
    "coverage": 0.918,
    "prompt_tokens": 262428.0,
    "output_tokens": 58363.0,
    "est_cost_usd": 0.8313645000000001,
    "corr": {
      "has_text_in_image": {
        "rate": -0.01897564083181411,
        "log_eng": -0.02034547131827701,
        "mean": 0.7254901960784313
      },
      "is_screenshot": {
        "rate": -0.021719496319151318,
        "log_eng": -0.023182287650287712,
        "mean": 0.4074074074074074
      },
      "is_meme": {
        "rate": 0.03878865776114655,
        "log_eng": 0.03876911174906618,
        "mean": 0.1721132897603486
      },
      "is_chart": {
        "rate": -0.0330865686313267,
        "log_eng": -0.0334174599124425,
        "mean": 0.06753812636165578
      },
      "has_face": {
        "rate": 0.014203396834723964,
        "log_eng": 0.016343440246768256,
        "mean": 0.3681917211328976
      },
      "is_photo_realworld": {
        "rate": 0.09023753613803627,
        "log_eng": 0.09193960892030771,
        "mean": 0.4074074074074074
      },
      "humor_0_10": {
        "rate": 0.06483441850487859,
        "log_eng": 0.06595760393690049,
        "mean": 1.7821350762527233
      },
      "controversy_0_10": {
        "rate": 0.09301966306001913,
        "log_eng": 0.09484549754843698,
        "mean": 0.9324618736383442
      }
    },
    "linear_combo_features": [
      "has_text_in_image",
      "is_screenshot",
      "is_meme",
      "is_chart",
      "has_face",
      "is_photo_realworld",
      "humor_0_10",
      "controversy_0_10"
    ],
    "cv_ridge_spearman_rate": 0.02790314463152408,
    "insample_ridge_spearman_rate": 0.1097045406400848
  },
  "siglip": {
    "n_ok": 459,
    "coverage": 0.918,
    "dims": 768,
    "pc1_spearman_rate": -0.03349404957095756,
    "cv_ridge_spearman_rate": 0.06744001562536418,
    "cv_pca8_ridge_spearman_rate": 0.10062373221043282,
    "best_dim_spearman_rate": 0.15386260891273773
  }
}
```
