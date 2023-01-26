from typing import Tuple, Iterable
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from kaldi_io import read_vec_flt_scp
from timit import read_spkrinfo
from common import PathLike


# helper functions
def read_vectors(scp_file: PathLike) -> Tuple[Tuple[str], Tuple[np.ndarray]]:
    "Read Kaldi vectors stored on disk"
    return zip(*read_vec_flt_scp(str(scp_file)))


def export_processed_embeddings(fname: PathLike, keys: Iterable[str],
                                embeddings: Iterable[np.ndarray]):
    data = dict(zip(keys, embeddings))
    np.savez(fname, **data)


class XVectorPipeline:
    "Pipeline for processing raw x-vectors"
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.lda = LinearDiscriminantAnalysis(n_components=n_components)
        self.speaker_embeddings = None
        self.mu = np.zeros(n_components)
        self.sigma = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        n_speakers = len(np.unique(y))
        # labels have to be integers from 0 to n_speakers - 1
        assert n_speakers == max(y) + 1, "Invalid labels"

        # dimensionality reduction
        self.lda.fit(X, y)
        X_compressed = self.lda.transform(X)

        # get speaker embeddings by averaging
        self.speaker_embeddings = np.zeros((n_speakers, self.n_components))
        for i in range(n_speakers):
            mask = (y == i)
            self.speaker_embeddings[i, :] = np.mean(X_compressed[mask], axis=0)

        # normalize speaker embeddings
        self.mu = np.mean(self.speaker_embeddings, axis=0)
        self.speaker_embeddings -= self.mu
        norms = np.linalg.norm(self.speaker_embeddings, ord=2, axis=1)
        self.sigma = np.mean(norms)
        self.speaker_embeddings /= self.sigma

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self.sigma is not None, "Perform `fit` first"
        out = self.lda.transform(X)
        out -= self.mu
        return out / self.sigma


if __name__ == "__main__":
    from pathlib import Path

    # map speaker ids to integers
    spkrinfo = read_spkrinfo("data/TIMIT/DOC/SPKRINFO.TXT")
    spkrs = np.array(spkrinfo.index)
    spkr2token = {spkr: i for i, spkr in enumerate(spkrs)}

    pipeline = XVectorPipeline(n_components=128)

    # train subset (fit + transform)
    keys, embeddings = read_vectors("data/xvectors_train/xvector.scp")
    y = np.array([spkr2token[k.split("_")[0]] for k in keys])
    X = np.stack(embeddings)
    pipeline.fit(X, y)
    X_processed = pipeline.transform(X)

    export_processed_embeddings("data/xvectors_train/xvector.npz",
                                keys, X_processed)
    export_processed_embeddings("data/xvectors_train/spk_xvector.npz",
                                spkrs, pipeline.speaker_embeddings)

    # test subset (transform)
    for subset in ["test", "words"]:
        data_dir = Path(f"data/xvectors_{subset}")
        keys, embeddings = read_vectors(data_dir / "xvector.scp")
        X_processed = pipeline.transform(np.stack(embeddings))
        export_processed_embeddings(data_dir / "xvector.npz",
                                    keys, X_processed)
