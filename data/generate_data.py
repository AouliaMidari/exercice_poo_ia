import csv
import random
import os

def generate_data():
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", "patient_data.csv")
    
    with open(filepath, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["patient_id", "temperature", "toux", "fatigue", "age"])
        
        for i in range(1, 6):
            writer.writerow([f"P00{i}", round(random.uniform(36.0, 39.5), 1), random.choice([0,1]), random.choice([0,1]), random.randint(20,70)])
            
    print("✅ Fichier 'data/patient_data.csv' généré.")

if __name__ == "__main__":
    generate_data()