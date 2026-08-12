import os
import multiprocessing

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

from src import create_app

app = create_app()

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    app.run(host="0.0.0.0", port=8000, debug=True, use_reloader=False)