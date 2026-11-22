# Live melanoma screening demo

The demo runs the four-model comparison locally on three selected dermoscopy images. It shows the input image, age, sex, lesion site, and the three branch scores. The primary model averages the branch scores and marks the image positive when the average reaches the frozen threshold of `0.0135909608307573`. Inference resizes images proportionally and uses fixed gray padding to reach 224 × 224.

[One-slide demo introduction (PDF)](Fusion-intro.pdf) · [Final evaluation](../outputs/final_internal_test/summary_en.md)

## Run on the presentation computer

Use the team's full `Melanoma-Fusion-Demo.zip`. Extract the entire archive into a writable directory and keep the `assets` folder with the model, case configuration, and three images. We recommend 64-bit Python 3.12. The first launch installs dependencies online. Once installed, the demo runs offline on a CPU.

- **Windows:** Double-click `Start-Windows.bat`.
- **macOS or Linux:** Run `python3 launch.py` in the extracted directory.
- **Existing PyTorch, NumPy, and Pillow environment:** Run `python demo.py`.

Keep the terminal open during the demo. If the browser does not open automatically, use the local address printed in the terminal. The public repository contains the source and intro slide. The team's archive supplies the runtime assets and the backup recording.

## Presentation sequence

Click **Run fusion**, explain the three scores and the fixed threshold, then click **Reveal label**. Open **Compare models** to show all four predictions side by side.

| Case | What to show |
| --- | --- |
| **A** | The fusion model correctly flags a melanoma image. |
| **B** | The image is benign. The three comparators flag it, while the fusion model correctly leaves it unflagged. |
| **C** | The image is melanoma. The fusion score falls just below the fixed threshold, producing a false negative. |

[Chinese demo guide](README.zh.md)
