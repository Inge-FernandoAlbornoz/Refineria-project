from __future__ import annotations
from abc import ABC, abstractmethod
import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


# =====================================================================
# 1. CONTRATO Y MARCAS DE VDF (Principio Abierto/Cerrado - OCP)
# =====================================================================
class VariadorFrecuencia(ABC):
    def __init__(self):
        self.frecuencia_actual_hz = 0.0

    @abstractmethod
    def ajustar_frecuencia(self, hz: float):
        pass


class VdfDanfoss(VariadorFrecuencia):
    def ajustar_frecuencia(self, hz: float):
        self.frecuencia_actual_hz = hz


class VdfWeg(VariadorFrecuencia):
    def ajustar_frecuencia(self, hz: float):
        self.frecuencia_actual_hz = hz


# =====================================================================
# 2. PROTECCIÓN DE TABLERO, INSTRUMENTACIÓN Y BASE DE DATOS SQLITE (SRP)
# =====================================================================
class ProteccionTablero:
    """Simula Guardamotor y Contactor Principal."""
    def __init__(self, corriente_nominal_amp: float):
        self.limite_termico_amp = corriente_nominal_amp
        self.guardamotor_disparado = False
        self.contactor_cerrado = True

    def evaluar_corriente(self, corriente_actual: float):
        if corriente_actual > self.limite_termico_amp:
            self.guardamotor_disparado = True
            self.contactor_cerrado = False

    def rearmar_manual(self):
        self.guardamotor_disparado = False
        self.contactor_cerrado = True

    @property
    def estado_contactor(self) -> str:
        return "PEGADO (ON)" if self.contactor_cerrado else "ABIERTO (OFF)"


class TransmisorPresion:
    def __init__(self):
        self.presion_actual = 4.2

    def actualizar_lectura(self, nueva_presion: float):
        self.presion_actual = nueva_presion

    def obtener_presion_bar(self) -> float:
        return self.presion_actual


class Electrovalvula:
    def __init__(self):
        self.abierta = False

    def abrir(self):
        self.abierta = True

    def cerrar(self):
        self.abierta = False

    @property
    def estado_texto(self) -> str:
        return "ABIERTA" if self.abierta else "CERRADA"


class SistemaLogSCADA:
    """Historiador SCADA conectado a SQLite."""
    def __init__(self, nombre_bd: str = "planta_oilgas.db"):
        self.nombre_bd = nombre_bd
        self.historial_registros = list()
        self._inicializar_tabla_sql()

    def _inicializar_tabla_sql(self):
        conexion = sqlite3.connect(self.nombre_bd)
        cursor = conexion.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS telemetria_bomba (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hora INTEGER,
                temperatura_c REAL,
                vibracion_mms REAL,
                corriente_amp REAL,
                presion_bar REAL,
                frecuencia_vdf_hz REAL,
                contactor TEXT,
                electrovalvula TEXT,
                diagnostico_ia TEXT,
                accion_plc TEXT
            )
        """)
        conexion.commit()
        conexion.close()

    def registrar_ciclo(self, datos: dict):
        self.historial_registros.append(datos)
        conexion = sqlite3.connect(self.nombre_bd)
        cursor = conexion.cursor()
        cursor.execute("""
            INSERT INTO telemetria_bomba (
                hora, temperatura_c, vibracion_mms, corriente_amp, presion_bar,
                frecuencia_vdf_hz, contactor, electrovalvula, diagnostico_ia, accion_plc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datos.get("Hora"),
            datos.get("Temperatura_C"),
            datos.get("Vibracion_mms"),
            datos.get("Corriente_Amp"),
            datos.get("Presion_Bar"),
            datos.get("Frecuencia_VDF_Hz"),
            datos.get("Contactor"),
            datos.get("Electrovalvula"),
            datos.get("Diagnostico_IA"),
            datos.get("Accion_PLC")
        ))
        conexion.commit()
        conexion.close()
        
    def exportar_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.historial_registros)
    
    
class BombaDesplazamientoPositivo:
    def __init__(self, tag_equipo: str):
        self.tag_equipo = tag_equipo
        self.temperatura_carcasa = 68.0
        self.vibracion_rms = 2.1
        self.corriente_amp = 32.0
        self.horas_operacion = 3000

    def actualizar_telemetria(self, temp: float, vib: float, amp: float, horas: int):
        self.temperatura_carcasa = temp
        self.vibracion_rms = vib
        self.corriente_amp = amp
        self.horas_operacion = horas

    def leer_telemetria(self) -> dict:
        return {
            "tag": self.tag_equipo,
            "temperatura": self.temperatura_carcasa,
            "vibracion": self.vibracion_rms,
            "corriente": self.corriente_amp,
            "horas": self.horas_operacion
        }
        
    
class PredictorFallasML:
    def __init__(self):
        self.modelo = RandomForestClassifier(n_estimators=100, random_state=42)
        self.modelo_entrenado = False
        self.etiquetas_diagnostico = {
            0: "Óptimo",
            1: "Alerta Mecánica",
            2: "Alerta Térmica",
            3: "Crítico"
        }

    def entrenar_con_historico(self, X_caracteristicas: list, y_estados: list):
        self.modelo.fit(np.array(X_caracteristicas), np.array(y_estados))
        self.modelo_entrenado = True

    def diagnosticar(self, telemetria: dict):
        fila = (telemetria.get("temperatura"), telemetria.get("vibracion"), telemetria.get("horas"))
        entrada = np.array(list((fila,)))
        prediccion = int(self.modelo.predict(entrada).item())
        etiqueta = self.etiquetas_diagnostico.get(prediccion)
        hay_riesgo = prediccion != 0
        return hay_riesgo, etiqueta
            
# =====================================================================
# 4. CEREBRO DE CONTROL (PLC)
# =====================================================================

class ControladorProcesoPLC:
    def __init__(self, proteccion: ProteccionTablero, transmisor: TransmisorPresion,
                 vdf: VariadorFrecuencia, valvula: Electrovalvula,
                 bomba: BombaDesplazamientoPositivo, predictor: PredictorFallasML,
                 scada: SistemaLogSCADA):
        self.proteccion = proteccion
        self.transmisor = transmisor
        self.vdf = vdf
        self.valvula = valvula
        self.bomba = bomba
        self.predictor = predictor
        self.scada = scada

    def ejecutar_ciclo_control(self, numero_ciclo: int):
        datos_bomba = self.bomba.leer_telemetria()
        presion = self.transmisor.obtener_presion_bar()
        
        self.proteccion.evaluar_corriente(datos_bomba.get("corriente"))
        hay_riesgo_mecanico, diagnostico_ia = self.predictor.diagnosticar(datos_bomba)

        if self.proteccion.guardamotor_disparado:
            self.vdf.ajustar_frecuencia(0.0)
            self.valvula.abrir()
            accion_plc = "DISPARO GUARDAMOTOR (0 Hz)"
        elif hay_riesgo_mecanico:
            self.vdf.ajustar_frecuencia(20.0)
            self.valvula.abrir()
            accion_plc = "Protección Predictiva IA (20 Hz)"
        elif presion > 5.0:
            self.vdf.ajustar_frecuencia(30.0)
            self.valvula.abrir()
            accion_plc = "Alivio por Presión (30 Hz)"
        else:
            self.vdf.ajustar_frecuencia(50.0)
            self.valvula.cerrar()
            accion_plc = "Régimen Normal (50 Hz)"

        self.scada.registrar_ciclo({
            "Hora": numero_ciclo,
            "Temperatura_C": datos_bomba.get("temperatura"),
            "Vibracion_mms": datos_bomba.get("vibracion"),
            "Corriente_Amp": datos_bomba.get("corriente"),
            "Presion_Bar": presion,
            "Frecuencia_VDF_Hz": self.vdf.frecuencia_actual_hz,
            "Contactor": self.proteccion.estado_contactor,
            "Electrovalvula": self.valvula.estado_texto,
            "Diagnostico_IA": diagnostico_ia,
            "Accion_PLC": accion_plc
        })


# =====================================================================
# 5. VISUALIZACIÓN Y TABLERO SCADA (SRP - Solo gráficos)
# =====================================================================
class TableroGraficoSCADA:
    def generar_tablero(self, df: pd.DataFrame):
        fig, axes = plt.subplots(2, 3, figsize=(17, 9))
        fig.suptitle("Tablero Integrado SCADA + PLC + Tablero Eléctrico + IA: Bomba P-101A", fontsize=14, fontweight="bold")

        # --- GRÁFICO 1: Dispersión (Temp vs Vibración) ---
        ax1 = axes[0, 0]
        optimos = df[df["Diagnostico_IA"] == "Óptimo"]
        alertas = df[df["Diagnostico_IA"] != "Óptimo"]
        ax1.scatter(optimos["Temperatura_C"], optimos["Vibracion_mms"], color="green", label="Óptimo (IA)", alpha=0.7)
        ax1.scatter(alertas["Temperatura_C"], alertas["Vibracion_mms"], color="red", label="Falla / Outlier", s=80, edgecolors="black")
        ax1.axhline(y=4.0, color="orange", linestyle="--", label="Límite Vib (4.0 mm/s)")
        ax1.set_title("1. Dispersión: Temp vs Acelerómetro")
        ax1.set_xlabel("Temperatura (°C)")
        ax1.set_ylabel("Vibración (mm/s)")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)

        # --- GRÁFICO 2: Corriente del Motor vs Límite Guardamotor ---
        ax2 = axes[0, 1]
        ax2.plot(df["Hora"], df["Corriente_Amp"], color="darkorange", linewidth=1.5, label="Consumo Motor (A)")
        ax2.axhline(y=45.0, color="red", linestyle="--", label="Corte Guardamotor (45 A)")
        ax2.set_title("2. Consumo Eléctrico y Disparo Térmico")
        ax2.set_xlabel("Hora de Operación")
        ax2.set_ylabel("Corriente (Amperes)")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

        # --- GRÁFICO 3: Caja y Bigotes (Acelerómetro) ---
        ax3 = axes [0, 2]
        ax3.boxplot(df["Vibracion_mms"], patch_artist=True, boxprops=dict(facecolor="lightblue"))
        ax3.set_title("3. Bigotes: Estabilidad del Acelerómetro")
        ax3.set_ylabel("Vibración (mm/s)")
        ax3.set_xticklabels(["Sensor Acelerómetro"])
        ax3.grid(True, linestyle=":", alpha=0.6)

        # --- GRÁFICO 4: Serie Temporal Presión de Línea ---
        ax4 = axes[1, 0]
        ax4.plot(df["Hora"], df["Presion_Bar"], color="navy", linewidth=1.5, label="Presión Línea")
        ax4.axhline(y=5.0, color="red", linestyle="--", label="SetPoint Alivio (5.0 bar)")
        ax4.set_title("4. Transmisor de Presión (100 Horas)")
        ax4.set_xlabel("Hora de Operación")
        ax4.set_ylabel("Presión (bar)")
        ax4.legend()
        ax4.grid(True, linestyle=":", alpha=0.6)

        # --- GRÁFICO 5: Respuesta de Frecuencia del VDF WEG ---
        ax5 = axes[1, 1]
        ax5.step(df["Hora"], df["Frecuencia_VDF_Hz"], color="purple", where="mid", linewidth=1.8)
        ax5.set_title("5. Consigna del PLC al VDF WEG (Hz)")
        ax5.set_xlabel("Hora de Operación")
        ax5.set_ylabel("Frecuencia (Hz)")
        ax5.set_ylim(-2, 60)
        ax5.grid(True, linestyle=":", alpha= 0.6)
        
        # --- GRÁFICO 6: Barras de Acciones del PLC ---
        ax6 = axes[1, 2]
        conteo_acciones = df["Accion_PLC"].value_counts()
        ax6.barh(conteo_acciones.index, conteo_acciones.values, color="teal")
        ax6.set_title("6. Resumen de Decisiones del PLC")
        ax6.set_xlabel("Horas en cada modo")
        ax6.grid(axis="x", linestyle=":", alpha=0.6)

        plt.tight_layout()
        plt.show()
        
#=====================================================================
# 6. SIMULACIÓN DE "DESPIOLE" EN PLANTA + GUARDADO EN SQLITE
# =====================================================================
if __name__ == "__main__":
    proteccion = ProteccionTablero(corriente_nominal_amp=45.0)
    transmisor = TransmisorPresion()
    vdf = VdfWeg()
    valvula = Electrovalvula()
    bomba = BombaDesplazamientoPositivo(tag_equipo="P-101A")
    scada = SistemaLogSCADA(nombre_bd="planta_oilgas.db")
    predictor = PredictorFallasML()

    X_historico = [
        [65.0, 1.8, 1200], [68.0, 2.0, 2500], [70.0, 2.2, 3100],
        [72.0, 4.5, 4000], [75.0, 5.1, 4500],
        [88.0, 2.3, 3800], [92.0, 2.5, 4200],
        [95.0, 5.8, 5000], [98.0, 6.2, 5500]
    ]
    y_estados = [0, 0, 0, 1, 1, 2, 2, 3, 3]
    predictor.entrenar_con_historico(X_historico, y_estados)

    plc = ControladorProcesoPLC(proteccion, transmisor, vdf, valvula, bomba,predictor, scada)
    np.random.seed(42)
    for hora in range(1, 101):
        temp_sim = np.random.normal(68.0, 1.5)
        vib_sim = np.random.normal(2.1, 0.2)
        amp_sim = np.random.normal(32.0, 1.2)
        presion_sim = np.random.normal(4.2, 0.15)

        if 15 <= hora <= 25:
            presion_sim = np.random.uniform(5.4, 6.1)
            amp_sim = 39.0
        elif 40 <= hora <= 55:
            vib_sim = np.random.uniform(4.8, 6.5)
            temp_sim = np.random.uniform(78.0, 85.0)
        elif 65 <= hora <= 75:
            temp_sim = 96.0
            vib_sim = 6.2
            presion_sim = 5.9
            amp_sim = 52.0
        elif hora == 76:
            proteccion.rearmar_manual()
        elif 85 <= hora <= 92:
            presion_sim = 5.5
            
        transmisor.actualizar_lectura(round(presion_sim, 2))
        bomba.actualizar_telemetria(round(temp_sim, 1), round(vib_sim, 2), round(amp_sim, 1), 3000 + hora)
        plc.ejecutar_ciclo_control(hora)

    print("Simulación finalizada. Datos guardados en 'planta_oilgas.db'.")
    df_resultados = scada.exportar_dataframe()
    print(df_resultados.iloc[63:78])

    tablero = TableroGraficoSCADA()
    tablero.generar_tablero(df_resultados) 