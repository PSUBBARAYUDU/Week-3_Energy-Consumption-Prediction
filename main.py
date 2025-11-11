from src.data_preprocessing import load_and_preprocess_data
from src.model_training import train_model
from src.prediction import predict_energy
from src.visualization import plot_relationship

def main():
    data = load_and_preprocess_data('data/energy_sample_data_2025.csv')
    plot_relationship(data, 'Temperature', 'Energy_Consumption')
    model = train_model(data, target_col='Energy_Consumption')
    print("Prediction Example:", predict_energy(model, [25]))

if __name__ == "__main__":
    main()
