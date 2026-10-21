# Edge Detection Tool

Interactive computer vision project comparing **Canny**, **Sobel**, and **Laplacian** edge detection. Upload a PNG/JPEG, tune parameters, compare the input with the result, and download the edge map. A CLI supports repeatable experiments.

## Setup and run

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

On Windows, activate with `.venv\Scripts\activate`. Open the local URL shown by Streamlit. No account, API key, or dataset is required.

```bash
python edge_detection.py input.jpg output.png --method canny --low 50 --high 150
python edge_detection.py input.jpg sobel.png --method sobel --kernel 3 --blur 3
python edge_detection.py input.jpg laplacian.png --method laplacian --kernel 3 --blur 3
python -m unittest discover -s tests -v
```

Run `python edge_detection.py --help` for all options. Outputs are 8-bit grayscale PNG images. See [RESEARCH.md](RESEARCH.md) for theory, evaluation, and limitations.
