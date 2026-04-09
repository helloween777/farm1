# update_pipeline.py
# Orquestador de ejecución de todas las capas (Bronze → Silver → Gold → ML)

import subprocess
import sys
import os


def run_script(script_name):
    """Ejecuta un script de Python en la carpeta src/"""
    print(f"\n[INFO] Ejecutando {script_name}...")
    print("=" * 60)
    
    script_path = os.path.join("src", script_name)
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    print(result.stdout)
    
    if result.stderr:
        print("[ERROR]", result.stderr)
        return False
    
    return True


def main():
    print("=" * 60)
    print("[INFO] INICIANDO PIPELINE COMPLETO")
    print("[INFO] Bronze → Silver → Gold → Entrenamiento")
    print("=" * 60)
    
    # 1. Capa Bronze (carga de CSVs)
    if not run_script("bronze.py"):
        print("[ERROR] Falló Bronze. Deteniendo pipeline.")
        sys.exit(1)
    
    # 2. Capa Silver (procesamiento y features)
    if not run_script("silver.py"):
        print("[ERROR] Falló Silver. Deteniendo pipeline.")
        sys.exit(1)
    
    # 3. Capa Gold (ML training data y alertas)
    if not run_script("gold.py"):
        print("[ERROR] Falló Gold. Deteniendo pipeline.")
        sys.exit(1)
    
    # 4. Entrenamiento de modelos (opcional, puede fallar sin detener)
    print("\n[INFO] Entrenando modelos (esto puede tomar varios minutos)...")
    run_script("train_models.py")
    
    print("\n" + "=" * 60)
    print("[INFO] ✅ PIPELINE COMPLETADO")
    print("[INFO] Tablas actualizadas: bronze.*, silver.*, gold.*")
    print("[INFO] Modelos y predicciones generados")
    print("=" * 60)


if __name__ == "__main__":
    main()