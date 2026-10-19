# core/dataset.py
import csv
import os


class Dataset:
    """
    Classe responsable du chargement et de la préparation des données.
    Respecte le SRP : elle ne fait QUE gérer les données (pas d'entraînement).
    """

    def __init__(self, filepath="data/patient_data.csv"):
        self.filepath = filepath
        self.X = []  # features (température, toux, fatigue, âge)
        self.y = []  # labels (0 = Sain, 1 = Infecté)

    def load(self):
        """Charge le CSV et sépare X (features) et y (labels simulés)."""
        if not os.path.exists(self.filepath):
            print(f"  [Dataset] ⚠️  Fichier introuvable : {self.filepath}")
            return self

        with open(self.filepath, mode="r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Features : température, toux, fatigue, âge
                self.X.append([
                    float(row["temperature"]),
                    int(row["toux"]),
                    int(row["fatigue"]),
                    int(row["age"]),
                ])
                # Label simulé : infecté si température > 38.0
                self.y.append(1 if float(row["temperature"]) > 38.0 else 0)

        print(f"  [Dataset] ✅ {len(self.X)} patients chargés depuis '{self.filepath}'")
        return self

    def split(self, ratio=0.8):
        """Séparation train / test simple (utile plus tard)."""
        n = int(len(self.X) * ratio)
        print(f"  [Dataset] Split : {n} train / {len(self.X) - n} test")
        return (self.X[:n], self.y[:n]), (self.X[n:], self.y[n:])

    def __len__(self):
        return len(self.X)

    def __repr__(self):
        return f"Dataset(n_samples={len(self.X)})"