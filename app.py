import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from scipy import stats
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional

# ========================================
# CLASE BASE PARA ANÁLISIS DE DATOS
# ========================================
class DataAnalyzer:
    """Clase para realizar análisis exploratorio de datos"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    
    def get_basic_info(self) -> Dict:
        """Retorna información básica del dataset"""
        return {
            'n_rows': self.df.shape[0],
            'n_cols': self.df.shape[1],
            'null_values': self.df.isnull().sum().sum(),
            'duplicates': self.df.duplicated().sum()
        }
    
    def get_data_types_info(self) -> pd.DataFrame:
        """Retorna información sobre tipos de datos"""
        return pd.DataFrame({
            'Columna': self.df.columns,
            'Tipo de Dato': self.df.dtypes.astype(str).values,
            'Valores Nulos': self.df.isnull().sum().values,
            '% Nulos': (self.df.isnull().sum().values / len(self.df) * 100).round(2),
            'Valores Únicos': [self.df[col].nunique() for col in self.df.columns]
        })
    
    def get_descriptive_stats(self) -> pd.DataFrame:
        """Retorna estadísticas descriptivas completas"""
        if len(self.numeric_columns) == 0:
            return pd.DataFrame()
        
        stats_df = self.df[self.numeric_columns].describe().T
        stats_df['Varianza'] = self.df[self.numeric_columns].var()
        stats_df['Asimetría (Skewness)'] = self.df[self.numeric_columns].skew()
        stats_df['Curtosis'] = self.df[self.numeric_columns].kurtosis()
        stats_df['Coef. Variación'] = (
            self.df[self.numeric_columns].std() / 
            self.df[self.numeric_columns].mean() * 100
        ).round(2)
        return stats_df.round(3)
    
    def detect_outliers(self, column: str) -> Dict:
        """Detecta outliers usando el método IQR"""
        Q1 = self.df[column].quantile(0.25)
        Q3 = self.df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
        
        return {
            'count': outliers_mask.sum(),
            'percentage': (outliers_mask.sum() / len(self.df) * 100),
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'mask': outliers_mask
        }
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Retorna matriz de correlación"""
        if len(self.numeric_columns) < 2:
            return pd.DataFrame()
        return self.df[self.numeric_columns].corr()
    
    def get_strong_correlations(self, threshold: float = 0.5) -> List[Dict]:
        """Encuentra correlaciones fuertes entre variables"""
        corr = self.get_correlation_matrix()
        if corr.empty:
            return []
        
        correlations = []
        for i in range(len(corr.columns)):
            for j in range(i+1, len(corr.columns)):
                if abs(corr.iloc[i, j]) > threshold:
                    correlations.append({
                        'Variable 1': corr.columns[i],
                        'Variable 2': corr.columns[j],
                        'Correlación': corr.iloc[i, j]
                    })
        
        return sorted(correlations, key=lambda x: abs(x['Correlación']), reverse=True)


# ========================================
# CLASE PARA LIMPIEZA DE DATOS
# ========================================
class DataCleaner:
    """Clase para limpieza y preprocesamiento de datos"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
    
    def handle_missing_values(self, method: str) -> pd.DataFrame:
        """Maneja valores nulos según el método especificado"""
        df_clean = self.df.copy()
        
        if method == "drop":
            return df_clean.dropna()
        
        elif method == "mean":
            for col in df_clean.select_dtypes(include=np.number).columns:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
        
        elif method == "median":
            for col in df_clean.select_dtypes(include=np.number).columns:
                df_clean[col].fillna(df_clean[col].median(), inplace=True)
        
        elif method == "mode":
            for col in df_clean.columns:
                if not df_clean[col].mode().empty:
                    df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        elif method == "interpolate":
            df_clean = df_clean.interpolate(method='linear', limit_direction='both')
        
        return df_clean
    
    def remove_duplicates(self) -> pd.DataFrame:
        """Elimina filas duplicadas"""
        return self.df.drop_duplicates()
    
    def handle_outliers(self, column: str, method: str, bounds: Dict) -> pd.DataFrame:
        """Maneja outliers según el método especificado"""
        df_clean = self.df.copy()
        
        if method == "drop":
            mask = (df_clean[column] >= bounds['lower_bound']) & \
                   (df_clean[column] <= bounds['upper_bound'])
            return df_clean[mask]
        
        elif method == "cap":
            df_clean[column] = df_clean[column].clip(
                bounds['lower_bound'], 
                bounds['upper_bound']
            )
        
        return df_clean
    
    def transform_column(self, column: str, method: str) -> pd.DataFrame:
        """Aplica transformación a una columna"""
        df_clean = self.df.copy()
        
        if method == "normalize":
            min_val = df_clean[column].min()
            max_val = df_clean[column].max()
            df_clean[column] = (df_clean[column] - min_val) / (max_val - min_val)
        
        elif method == "standardize":
            mean_val = df_clean[column].mean()
            std_val = df_clean[column].std()
            df_clean[column] = (df_clean[column] - mean_val) / std_val
        
        elif method == "log":
            if (df_clean[column] < 0).any():
                raise ValueError("No se puede aplicar log a valores negativos")
            df_clean[column] = np.log1p(df_clean[column])
        
        elif method == "sqrt":
            if (df_clean[column] < 0).any():
                raise ValueError("No se puede aplicar sqrt a valores negativos")
            df_clean[column] = np.sqrt(df_clean[column])
        
        elif method == "boxcox":
            if (df_clean[column] <= 0).any():
                raise ValueError("Box-Cox requiere valores estrictamente positivos")
            transformed, lambda_param = stats.boxcox(df_clean[column])
            df_clean[column] = transformed
        
        return df_clean


# ========================================
# CLASE BASE PARA MODELOS
# ========================================
class MLModel(ABC):
    """Clase abstracta base para modelos de Machine Learning"""
    
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False
    
    @abstractmethod
    def create_model(self, **params):
        """Crea el modelo con parámetros específicos"""
        pass
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Entrena el modelo"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
    
    def predict(self, X_test: pd.DataFrame):
        """Realiza predicciones"""
        if not self.is_trained:
            raise Exception("El modelo no ha sido entrenado")
        return self.model.predict(X_test)
    
    @abstractmethod
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        """Retorna métricas de evaluación"""
        pass


# ========================================
# MODELOS DE REGRESIÓN
# ========================================
class LinearRegressionModel(MLModel):
    """Modelo de Regresión Lineal"""
    
    def __init__(self):
        super().__init__("Regresión Lineal")
    
    def create_model(self, **params):
        self.model = LinearRegression(**params)
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }
    
    def get_coefficients(self, feature_names: List[str]) -> pd.DataFrame:
        """Retorna los coeficientes del modelo"""
        return pd.DataFrame({
            'Variable': feature_names,
            'Coeficiente': self.model.coef_
        }).sort_values('Coeficiente', key=abs, ascending=False)


class RidgeRegressionModel(MLModel):
    """Modelo de Regresión Ridge"""
    
    def __init__(self):
        super().__init__("Ridge")
    
    def create_model(self, alpha=1.0, **params):
        self.model = Ridge(alpha=alpha, **params)
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }


class LassoRegressionModel(MLModel):
    """Modelo de Regresión Lasso"""
    
    def __init__(self):
        super().__init__("Lasso")
    
    def create_model(self, alpha=0.1, **params):
        self.model = Lasso(alpha=alpha, **params)
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }
    
    def get_coefficients(self, feature_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            'Variable': feature_names,
            'Coeficiente': self.model.coef_
        }).sort_values('Coeficiente', key=abs, ascending=False)


class DecisionTreeModel(MLModel):
    """Modelo de Árbol de Decisión"""
    
    def __init__(self):
        super().__init__("CART")
    
    def create_model(self, max_depth=4, random_state=42, **params):
        self.model = DecisionTreeRegressor(
            max_depth=max_depth, 
            random_state=random_state, 
            **params
        )
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        return pd.DataFrame({
            'Variable': feature_names,
            'Importancia': self.model.feature_importances_
        }).sort_values('Importancia', ascending=False)


class KNNModel(MLModel):
    """Modelo K-Nearest Neighbors"""
    
    def __init__(self):
        super().__init__("KNN")
    
    def create_model(self, n_neighbors=5, **params):
        self.model = KNeighborsRegressor(n_neighbors=n_neighbors, **params)
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }


class NeuralNetworkModel(MLModel):
    """Modelo de Red Neuronal"""
    
    def __init__(self):
        super().__init__("Red Neuronal")
    
    def create_model(self, hidden_layer_sizes=(50, 50), max_iter=1000, 
                     random_state=42, **params):
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            **params
        )
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'R²': r2_score(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'MAE': np.mean(np.abs(y_test - y_pred))
        }


class LogisticRegressionModel(MLModel):
    """Modelo de Regresión Logística"""
    
    def __init__(self):
        super().__init__("Regresión Logística")
    
    def create_model(self, max_iter=1000, **params):
        self.model = LogisticRegression(max_iter=max_iter, **params)
        return self
    
    def get_metrics(self, y_test: pd.Series, y_pred) -> Dict:
        return {
            'Accuracy': accuracy_score(y_test, y_pred),
            'Confusion_Matrix': confusion_matrix(y_test, y_pred)
        }


# ========================================
# CLASE PARA COMPARACIÓN DE MODELOS
# ========================================
class ModelComparator:
    """Clase para comparar múltiples modelos"""
    
    def __init__(self):
        self.models = []
        self.results = []
    
    def add_model(self, model: MLModel):
        """Agrega un modelo a la comparación"""
        self.models.append(model)
    
    def compare_models(self, X_train, X_test, y_train, y_test) -> pd.DataFrame:
        """Entrena y compara todos los modelos"""
        self.results = []
        
        for model in self.models:
            try:
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                metrics = model.get_metrics(y_test, y_pred)
                
                result = {'Modelo': model.name}
                result.update(metrics)
                self.results.append(result)
            except Exception as e:
                st.warning(f"Error entrenando {model.name}: {str(e)}")
        
        return pd.DataFrame(self.results)
    
    def get_best_model(self, metric='R²') -> Optional[str]:
        """Retorna el nombre del mejor modelo según una métrica"""
        if not self.results:
            return None
        
        df = pd.DataFrame(self.results)
        if metric in df.columns:
            return df.loc[df[metric].idxmax(), 'Modelo']
        return None


# ========================================
# CLASE PRINCIPAL DE LA APLICACIÓN
# ========================================
class MLDashboard:
    """Clase principal que maneja toda la aplicación"""
    
    def __init__(self):
        self.configure_page()
        self.df_original = None
        self.df_procesado = None
        self.analyzer = None
        self.cleaner = None
    
    def configure_page(self):
        """Configura la página de Streamlit"""
        st.set_page_config(
            page_title="Dashboard ML Completo",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        st.title("🤖 Dashboard Interactivo de Machine Learning con Análisis y Limpieza de Datos")
        st.markdown("---")
    
    def load_data(self) -> bool:
        """Carga los datos desde el archivo CSV"""
        st.sidebar.header("📂 Carga de Datos CSV")
        st.sidebar.markdown("Sube un archivo CSV con entre 500 y 1000 filas")
        archivo = st.sidebar.file_uploader("Selecciona tu archivo CSV", type=["csv"])
        
        if archivo is not None:
            self.df_original = pd.read_csv(archivo)
            
            if 'df_procesado' not in st.session_state:
                st.session_state.df_procesado = self.df_original.copy()
            
            self.df_procesado = st.session_state.df_procesado
            self.analyzer = DataAnalyzer(self.df_procesado)
            self.cleaner = DataCleaner(self.df_procesado)
            
            return True
        return False
    
    def validate_data(self) -> bool:
        """Valida que el dataset cumpla con los requisitos"""
        st.subheader("📋 Vista Previa de los Datos")
        st.dataframe(self.df_procesado.head(10), use_container_width=True)
        
        info = self.analyzer.get_basic_info()
        
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("📊 Total Filas", info['n_rows'])
        with col_m2:
            st.metric("📊 Total Columnas", info['n_cols'])
        with col_m3:
            st.metric("❌ Valores Nulos", info['null_values'])
        with col_m4:
            st.metric("🔁 Duplicados", info['duplicates'])
        
        if not (500 <= info['n_rows'] <= 1000):
            st.error(f"❌ **ERROR**: Solo se permiten datasets con entre 500 y 1000 filas.")
            st.info(f"Tu dataset tiene {info['n_rows']} filas. Por favor, ajusta tu archivo.")
            return False
        
        st.success(f"✅ Dataset válido: {info['n_rows']} filas × {info['n_cols']} columnas")
        st.markdown("---")
        return True
    
    def show_exploratory_analysis(self):
        """Muestra la pestaña de análisis exploratorio"""
        st.header("📊 Análisis Exploratorio de Datos (EDA)")
        st.markdown("Explora tu dataset en profundidad antes de aplicar modelos de ML")
        
        # Información general
        st.subheader("📌 1. Información General del Dataset")
        info = self.analyzer.get_basic_info()
        
        col_info1, col_info2, col_info3, col_info4 = st.columns(4)
        with col_info1:
            st.metric("Total Filas", info['n_rows'])
        with col_info2:
            st.metric("Total Columnas", info['n_cols'])
        with col_info3:
            st.metric("Valores Nulos", info['null_values'])
        with col_info4:
            st.metric("Filas Duplicadas", info['duplicates'])
        
        st.markdown("---")
        
        # Tipos de datos
        st.subheader("🔤 2. Tipos de Datos y Valores Nulos")
        tipos_df = self.analyzer.get_data_types_info()
        st.dataframe(tipos_df, use_container_width=True, height=300)
        
        st.markdown("---")
        
        # Estadísticas descriptivas
        st.subheader("📈 3. Estadísticas Descriptivas")
        stats_df = self.analyzer.get_descriptive_stats()
        
        if not stats_df.empty:
            st.dataframe(stats_df, use_container_width=True, height=400)
            st.info("""
            **📚 Interpretación de Estadísticas:**
            - **Asimetría**: Valores cercanos a 0 indican distribución simétrica
            - **Curtosis**: Valores cercanos a 0 = distribución normal
            - **Coef. Variación**: Mide dispersión relativa (%)
            """)
        else:
            st.warning("⚠️ No hay columnas numéricas en el dataset")
        
        st.markdown("---")
        
        # Visualizaciones
        if len(self.analyzer.numeric_columns) > 0:
            st.subheader("📉 4. Visualizaciones Interactivas")
            self._show_visualizations()
        
        st.markdown("---")
        
        # Matriz de correlación
        if len(self.analyzer.numeric_columns) >= 2:
            st.subheader("🔗 5. Matriz de Correlación")
            self._show_correlation_matrix()
    
    def _show_visualizations(self):
        """Muestra las visualizaciones del análisis exploratorio"""
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.markdown("**Histograma con Boxplot**")
            columna_hist = st.selectbox(
                "Selecciona columna",
                self.analyzer.numeric_columns,
                key="hist"
            )
            
            fig_hist = px.histogram(
                self.df_procesado,
                x=columna_hist,
                nbins=30,
                title=f"Distribución de {columna_hist}",
                marginal="box",
                color_discrete_sequence=['#636EFA']
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            col_stats = self.df_procesado[columna_hist].describe()
            st.write(f"**Media:** {col_stats['mean']:.2f} | **Mediana:** {col_stats['50%']:.2f} | **Desv. Std:** {col_stats['std']:.2f}")
        
        with viz_col2:
            if len(self.analyzer.numeric_columns) >= 2:
                st.markdown("**Gráfico de Dispersión**")
                x_col = st.selectbox("Eje X", self.analyzer.numeric_columns, index=0, key="scatter_x")
                y_col = st.selectbox("Eje Y", self.analyzer.numeric_columns, index=min(1, len(self.analyzer.numeric_columns)-1), key="scatter_y")
                
                fig_scatter = px.scatter(
                    self.df_procesado,
                    x=x_col,
                    y=y_col,
                    title=f"Relación: {x_col} vs {y_col}",
                    trendline="ols",
                    trendline_color_override="red"
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
                
                correlacion = self.df_procesado[x_col].corr(self.df_procesado[y_col])
                st.write(f"**Correlación de Pearson:** {correlacion:.3f}")
        
        # Detección de outliers
        st.markdown("---")
        st.markdown("#### 📦 Detección de Outliers")
        
        outlier_col1, outlier_col2 = st.columns([2, 1])
        
        with outlier_col1:
            columna_box = st.selectbox(
                "Selecciona columna para analizar outliers",
                self.analyzer.numeric_columns,
                key="box"
            )
            
            fig_box = px.box(
                self.df_procesado,
                y=columna_box,
                title=f"Boxplot de {columna_box}",
                color_discrete_sequence=['#EF553B']
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        with outlier_col2:
            outliers_info = self.analyzer.detect_outliers(columna_box)
            
            st.markdown("**📊 Estadísticas de Outliers:**")
            st.metric("Total Outliers", outliers_info['count'])
            st.metric("Porcentaje", f"{outliers_info['percentage']:.2f}%")
            st.write(f"**Q1:** {outliers_info['Q1']:.2f}")
            st.write(f"**Q3:** {outliers_info['Q3']:.2f}")
            st.write(f"**IQR:** {outliers_info['IQR']:.2f}")
            st.write(f"**Límite inferior:** {outliers_info['lower_bound']:.2f}")
            st.write(f"**Límite superior:** {outliers_info['upper_bound']:.2f}")
    
    def _show_correlation_matrix(self):
        """Muestra la matriz de correlación"""
        corr = self.analyzer.get_correlation_matrix()
        
        fig_corr = px.imshow(
            corr,
            text_auto='.2f',
            color_continuous_scale="RdBu_r",
            title="Matriz de Correlación entre Variables Numéricas",
            aspect="auto",
            zmin=-1,
            zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        st.markdown("**🔍 Correlaciones más fuertes (|r| > 0.5):**")
        strong_corr = self.analyzer.get_strong_correlations()
        
        if strong_corr:
            corr_df = pd.DataFrame(strong_corr)
            st.dataframe(corr_df, use_container_width=True)
        else:
            st.info("No se encontraron correlaciones fuertes entre las variables")
    
    def show_data_cleaning(self):
        """Muestra la pestaña de limpieza de datos"""
        st.header("🧹 Limpieza y Preprocesamiento de Datos")
        st.markdown("Aplica transformaciones y limpieza a tu dataset antes del modelado")
        
        # Botón de reset
        if st.button("🔄 Resetear a datos originales", use_container_width=True):
            st.session_state.df_procesado = self.df_original.copy()
            st.success("✅ Datos reseteados correctamente")
            st.rerun()
        
        st.markdown("---")
        
        # Valores nulos
        self._handle_missing_values()
        
        st.markdown("---")
        
        # Duplicados
        self._handle_duplicates()
        
        st.markdown("---")
        
        # Outliers
        self._handle_outliers()
        
        st.markdown("---")
        
        # Transformaciones
        self._handle_transformations()
        
        st.markdown("---")
        
        # Resumen
        self._show_cleaning_summary()
    
    def _handle_missing_values(self):
        """Maneja valores nulos"""
        st.subheader("❌ 1. Manejo de Valores Nulos")
        
        null_info = self.analyzer.get_data_types_info()
        null_info = null_info[null_info['Valores Nulos'] > 0]
        
        if len(null_info) > 0:
            st.warning(f"⚠️ Se encontraron valores nulos en {len(null_info)} columnas")
            st.dataframe(null_info, use_container_width=True)
            
            metodo_nulos = st.radio(
                "Selecciona método para manejar valores nulos:",
                [
                    "Eliminar filas con nulos",
                    "Rellenar con la media",
                    "Rellenar con la mediana",
                    "Rellenar con moda",
                    "Interpolación lineal"
                ]
            )
            
            if st.button("✅ Aplicar limpieza de nulos", type="primary"):
                method_map = {
                    "Eliminar filas con nulos": "drop",
                    "Rellenar con la media": "mean",
                    "Rellenar con la mediana": "median",
                    "Rellenar con moda": "mode",
                    "Interpolación lineal": "interpolate"
                }
                
                st.session_state.df_procesado = self.cleaner.handle_missing_values(
                    method_map[metodo_nulos]
                )
                st.success("✅ Valores nulos procesados correctamente")
                st.rerun()
        else:
            st.success("✅ No hay valores nulos en el dataset")
    
    def _handle_duplicates(self):
        """Maneja duplicados"""
        st.subheader("🔁 2. Eliminación de Duplicados")
        
        duplicados = self.df_procesado.duplicated().sum()
        
        col_dup1, col_dup2 = st.columns([1, 3])
        with col_dup1:
            st.metric("Filas duplicadas", duplicados)
        
        with col_dup2:
            if duplicados > 0:
                st.warning(f"⚠️ Se encontraron {duplicados} filas duplicadas")
                if st.button("🗑️ Eliminar duplicados", type="primary"):
                    st.session_state.df_procesado = self.cleaner.remove_duplicates()
                    st.success(f"✅ Se eliminaron {duplicados} filas duplicadas")
                    st.rerun()
            else:
                st.success("✅ No hay filas duplicadas")
    
    def _handle_outliers(self):
        """Maneja outliers"""
        st.subheader("🎯 3. Manejo de Outliers")
        
        if len(self.analyzer.numeric_columns) > 0:
            columna_outlier = st.selectbox(
                "Selecciona columna",
                self.analyzer.numeric_columns,
                key="outlier_col"
            )
            
            outliers_info = self.analyzer.detect_outliers(columna_outlier)
            
            col_out1, col_out2 = st.columns([2, 1])
            
            with col_out1:
                fig_out = go.Figure()
                fig_out.add_trace(go.Box(
                    y=self.df_procesado[columna_outlier],
                    name=columna_outlier,
                    marker_color='indianred'
                ))
                fig_out.update_layout(
                    title=f"Boxplot de {columna_outlier}",
                    showlegend=False
                )
                st.plotly_chart(fig_out, use_container_width=True)
            
            with col_out2:
                st.markdown("**📊 Estadísticas:**")
                st.metric("Outliers", outliers_info['count'])
                st.metric("Porcentaje", f"{outliers_info['percentage']:.2f}%")
                st.write(f"**Límite inferior:** {outliers_info['lower_bound']:.2f}")
                st.write(f"**Límite superior:** {outliers_info['upper_bound']:.2f}")
            
            metodo_outlier = st.radio(
                "Método de tratamiento:",
                ["Mantener outliers", "Eliminar outliers", "Aplicar capping"],
                key="metodo_outlier"
            )
            
            if metodo_outlier != "Mantener outliers":
                if st.button("🚀 Aplicar tratamiento", type="primary"):
                    method_map = {
                        "Eliminar outliers": "drop",
                        "Aplicar capping": "cap"
                    }
                    
                    st.session_state.df_procesado = self.cleaner.handle_outliers(
                        columna_outlier,
                        method_map[metodo_outlier],
                        outliers_info
                    )
                    st.success("✅ Tratamiento de outliers aplicado")
                    st.rerun()
    
    def _handle_transformations(self):
        """Maneja transformaciones"""
        st.subheader("🔄 4. Transformaciones de Variables")
        
        if len(self.analyzer.numeric_columns) > 0:
            col_trans1, col_trans2 = st.columns([2, 1])
            
            with col_trans1:
                columna_transform = st.selectbox(
                    "Columna a transformar",
                    self.analyzer.numeric_columns,
                    key="transform_col"
                )
                
                transformacion = st.selectbox(
                    "Tipo de transformación:",
                    [
                        "Ninguna",
                        "Normalización Min-Max (0-1)",
                        "Estandarización Z-score",
                        "Transformación Logarítmica",
                        "Transformación Raíz Cuadrada",
                        "Transformación Box-Cox"
                    ]
                )
            
            with col_trans2:
                st.markdown("**📚 Descripción:**")
                descriptions = {
                    "Normalización Min-Max (0-1)": "Escala valores entre 0 y 1",
                    "Estandarización Z-score": "Media 0 y desv. std. 1",
                    "Transformación Logarítmica": "Reduce asimetría positiva",
                    "Transformación Raíz Cuadrada": "Reduce varianza",
                    "Transformación Box-Cox": "Transforma a distribución normal"
                }
                st.info(descriptions.get(transformacion, "Sin transformación"))
            
            if transformacion != "Ninguna":
                if st.button("✨ Aplicar transformación", type="primary"):
                    method_map = {
                        "Normalización Min-Max (0-1)": "normalize",
                        "Estandarización Z-score": "standardize",
                        "Transformación Logarítmica": "log",
                        "Transformación Raíz Cuadrada": "sqrt",
                        "Transformación Box-Cox": "boxcox"
                    }
                    
                    try:
                        st.session_state.df_procesado = self.cleaner.transform_column(
                            columna_transform,
                            method_map[transformacion]
                        )
                        st.success(f"✅ Transformación aplicada a '{columna_transform}'")
                        st.rerun()
                    except ValueError as e:
                        st.error(f"❌ Error: {str(e)}")
    
    def _show_cleaning_summary(self):
        """Muestra resumen de cambios"""
        st.subheader("📋 5. Resumen de Cambios")
        
        col_res1, col_res2, col_res3 = st.columns(3)
        
        with col_res1:
            st.markdown("**📊 Dataset Original**")
            st.metric("Filas", self.df_original.shape[0])
            st.metric("Columnas", self.df_original.shape[1])
            st.metric("Nulos", self.df_original.isnull().sum().sum())
        
        with col_res2:
            st.markdown("**📊 Dataset Procesado**")
            st.metric("Filas", self.df_procesado.shape[0], 
                     delta=int(self.df_procesado.shape[0] - self.df_original.shape[0]))
            st.metric("Columnas", self.df_procesado.shape[1],
                     delta=int(self.df_procesado.shape[1] - self.df_original.shape[1]))
            st.metric("Nulos", self.df_procesado.isnull().sum().sum(),
                     delta=int(self.df_procesado.isnull().sum().sum() - self.df_original.isnull().sum().sum()))
        
        with col_res3:
            st.markdown("**📊 Cambios**")
            filas_eliminadas = self.df_original.shape[0] - self.df_procesado.shape[0]
            if filas_eliminadas != 0:
                st.write(f"✂️ Filas eliminadas: {abs(filas_eliminadas)}")
            else:
                st.info("Sin cambios")
        
        st.markdown("---")
        csv = self.df_procesado.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar CSV procesado",
            data=csv,
            file_name="dataset_procesado.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def show_model_training(self):
        """Muestra la pestaña de entrenamiento de modelos"""
        st.header("⚙️ Entrenar un Modelo Individual")
        
        st.sidebar.markdown("---")
        st.sidebar.header("⚙️ Configuración del Modelo")
        
        modelos_dict = {
            "Regresión Lineal": LinearRegressionModel,
            "Regresión Logística": LogisticRegressionModel,
            "Regresión Ridge": RidgeRegressionModel,
            "Regresión Lasso": LassoRegressionModel,
            "Árbol de Regresión (CART)": DecisionTreeModel,
            "K-Nearest Neighbors": KNNModel,
            "Red Neuronal (MLP)": NeuralNetworkModel
        }
        
        modelo_elegido = st.sidebar.selectbox("🤖 Selecciona un modelo", list(modelos_dict.keys()))
        
        # Configuración de variables
        target = st.sidebar.selectbox("🎯 Variable objetivo (y)", self.df_procesado.columns)
        features_disponibles = [c for c in self.df_procesado.columns if c != target]
        features = st.sidebar.multiselect(
            "📊 Variables predictoras (X)",
            features_disponibles,
            default=features_disponibles[:min(3, len(features_disponibles))]
        )
        
        # Parámetros
        st.sidebar.markdown("---")
        test_size = st.sidebar.slider("% Datos de prueba", 10, 50, 30) / 100
        random_state = st.sidebar.number_input("Semilla aleatoria", 0, 100, 42)
        
        # Parámetros específicos del modelo
        model_params = self._get_model_params(modelo_elegido)
        
        st.sidebar.markdown("---")
        entrenar = st.sidebar.button("🚀 ENTRENAR MODELO", type="primary", use_container_width=True)
        
        if entrenar:
            if len(features) == 0:
                st.error("❌ Selecciona al menos una variable predictora")
                return
            
            # Preparar datos
            X = self.df_procesado[features].select_dtypes(include=np.number)
            y = self.df_procesado[target]
            
            if X.shape[1] == 0:
                st.error("❌ Las variables predictoras deben ser numéricas")
                return
            
            # Limpiar nulos
            if X.isnull().sum().sum() > 0 or y.isnull().sum() > 0:
                st.warning("⚠️ Eliminando valores nulos...")
                X = X.dropna()
                y = y[X.index]
            
            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Entrenar modelo
            self._train_and_display_model(
                modelos_dict[modelo_elegido],
                modelo_elegido,
                model_params,
                X_train, X_test, y_train, y_test,
                X.columns
            )
    
    def _get_model_params(self, modelo_elegido: str) -> Dict:
        """Obtiene parámetros específicos del modelo"""
        params = {}
        
        if modelo_elegido == "Regresión Ridge":
            params['alpha'] = st.sidebar.slider("Alpha (regularización)", 0.01, 10.0, 1.0, 0.01)
        
        elif modelo_elegido == "Regresión Lasso":
            params['alpha'] = st.sidebar.slider("Alpha (regularización)", 0.001, 5.0, 0.1, 0.001)
        
        elif modelo_elegido == "Árbol de Regresión (CART)":
            params['max_depth'] = st.sidebar.slider("Profundidad máxima", 2, 20, 4)
        
        elif modelo_elegido == "K-Nearest Neighbors":
            params['n_neighbors'] = st.sidebar.slider("Número de vecinos (K)", 1, 20, 5)
        
        elif modelo_elegido == "Red Neuronal (MLP)":
            hidden_layers = st.sidebar.slider("Capas ocultas", 1, 3, 2)
            neurons = st.sidebar.slider("Neuronas por capa", 10, 100, 50, 10)
            params['hidden_layer_sizes'] = tuple([neurons] * hidden_layers)
        
        return params
    
    def _train_and_display_model(self, ModelClass, model_name, params, 
                                  X_train, X_test, y_train, y_test, feature_names):
        """Entrena y muestra resultados del modelo"""
        st.subheader(f"🔍 Resultados: {model_name}")
        
        # Información del split
        col_split1, col_split2, col_split3 = st.columns(3)
        with col_split1:
            st.metric("🔢 Total datos", len(X_train) + len(X_test))
        with col_split2:
            st.metric("📘 Datos entrenamiento", len(X_train))
        with col_split3:
            st.metric("📗 Datos prueba", len(X_test))
        
        st.markdown("---")
        
        with st.spinner(f"Entrenando {model_name}..."):
            try:
                # Crear y entrenar modelo
                model = ModelClass().create_model(**params)
                model.train(X_train, y_train)
                y_pred = model.predict(X_test)
                
                # Métricas
                metrics = model.get_metrics(y_test, y_pred)
                
                if 'R²' in metrics:
                    self._display_regression_results(
                        model, metrics, y_test, y_pred, feature_names
                    )
                else:
                    self._display_classification_results(metrics)
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
    
    def _display_regression_results(self, model, metrics, y_test, y_pred, feature_names):
        """Muestra resultados de regresión"""
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.metric("R² Score", f"{metrics['R²']:.4f}")
        with col_m2:
            st.metric("MSE", f"{metrics['MSE']:.4f}")
        with col_m3:
            st.metric("RMSE", f"{metrics['RMSE']:.4f}")
        with col_m4:
            st.metric("MAE", f"{metrics['MAE']:.4f}")
        
        # Interpretación
        if metrics['R²'] > 0.8:
            st.success("✅ Excelente ajuste del modelo")
        elif metrics['R²'] > 0.6:
            st.info("ℹ️ Buen ajuste del modelo")
        elif metrics['R²'] > 0.4:
            st.warning("⚠️ Ajuste moderado")
        else:
            st.error("❌ Ajuste pobre")
        
        # Gráficos
        col_graph1, col_graph2 = st.columns(2)
        
        with col_graph1:
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=y_test, y=y_pred,
                mode='markers',
                name='Predicciones',
                marker=dict(size=8, color='blue', opacity=0.6)
            ))
            fig_pred.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Línea perfecta',
                line=dict(color='red', dash='dash')
            ))
            fig_pred.update_layout(
                title="Valores Reales vs Predicciones",
                xaxis_title="Valores Reales",
                yaxis_title="Predicciones"
            )
            st.plotly_chart(fig_pred, use_container_width=True)
        
        with col_graph2:
            residuos = y_test - y_pred
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                x=y_pred, y=residuos,
                mode='markers',
                marker=dict(size=8, color='green', opacity=0.6)
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="red")
            fig_res.update_layout(
                title="Gráfico de Residuos",
                xaxis_title="Predicciones",
                yaxis_title="Residuos"
            )
            st.plotly_chart(fig_res, use_container_width=True)
        
        # Características adicionales según el modelo
        if hasattr(model, 'get_coefficients'):
            st.markdown("**📊 Coeficientes del Modelo:**")
            coef_df = model.get_coefficients(feature_names)
            st.dataframe(coef_df, use_container_width=True)
        
        if hasattr(model, 'get_feature_importance'):
            st.markdown("**📊 Importancia de Variables:**")
            feat_imp = model.get_feature_importance(feature_names)
            fig_imp = px.bar(feat_imp, x='Importancia', y='Variable', orientation='h')
            st.plotly_chart(fig_imp, use_container_width=True)
        
        if hasattr(model.model, 'loss_curve_'):
            st.markdown("**📈 Curva de Aprendizaje:**")
            fig_loss = px.line(
                y=model.model.loss_curve_,
                title="Pérdida durante entrenamiento",
                labels={'x': 'Iteración', 'y': 'Pérdida'}
            )
            st.plotly_chart(fig_loss, use_container_width=True)
        
        # Visualización de árbol
        if isinstance(model, DecisionTreeModel):
            st.markdown("**🌳 Visualización del Árbol:**")
            fig_tree, ax = plt.subplots(figsize=(20, 10))
            plot_tree(
                model.model,
                feature_names=feature_names,
                filled=True,
                fontsize=10,
                ax=ax,
                rounded=True
            )
            st.pyplot(fig_tree)
    
    def _display_classification_results(self, metrics):
        """Muestra resultados de clasificación"""
        st.metric("🎯 Exactitud (Accuracy)", f"{metrics['Accuracy']:.4f}")
        
        st.markdown("**🔢 Matriz de Confusión:**")
        fig_cm = px.imshow(
            metrics['Confusion_Matrix'],
            text_auto=True,
            title="Matriz de Confusión",
            labels=dict(x="Predicho", y="Real"),
            color_continuous_scale="Blues"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
    
    def show_model_comparison(self):
        """Muestra la pestaña de comparación de modelos"""
        st.header("📈 Comparación de Todos los Modelos")
        
        X_num = self.df_procesado.select_dtypes(include=np.number)
        
        if X_num.shape[1] < 2:
            st.error("❌ Se requieren al menos 2 columnas numéricas")
            return
        
        col_comp1, col_comp2 = st.columns([2, 1])
        
        with col_comp1:
            target_comp = st.selectbox("🎯 Variable objetivo", X_num.columns, key="target_comp")
        
        with col_comp2:
            test_size_comp = st.slider("% Datos de prueba", 10, 50, 30, key="test_comp") / 100
        
        comparar_btn = st.button("⚡ ENTRENAR Y COMPARAR", type="primary", use_container_width=True)
        
        if comparar_btn:
            with st.spinner("🔄 Entrenando modelos..."):
                X = X_num.drop(columns=[target_comp])
                y = X_num[target_comp]
                
                if X.isnull().sum().sum() > 0:
                    X = X.dropna()
                    y = y[X.index]
                
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size_comp, random_state=42
                )
                
                # Crear comparador
                comparator = ModelComparator()
                
                # Agregar modelos
                comparator.add_model(LinearRegressionModel().create_model())
                comparator.add_model(RidgeRegressionModel().create_model(alpha=1.0))
                comparator.add_model(LassoRegressionModel().create_model(alpha=0.1))
                comparator.add_model(DecisionTreeModel().create_model(max_depth=4))
                comparator.add_model(KNNModel().create_model(n_neighbors=5))
                comparator.add_model(NeuralNetworkModel().create_model(
                    hidden_layer_sizes=(50, 50)
                ))
                
                # Comparar
                progress_bar = st.progress(0)
                res_df = comparator.compare_models(X_train, X_test, y_train, y_test)
                progress_bar.progress(100)
                progress_bar.empty()
                
                if not res_df.empty:
                    self._display_comparison_results(res_df)
    
    def _display_comparison_results(self, res_df: pd.DataFrame):
        """Muestra resultados de comparación"""
        st.subheader("📊 Tabla de Resultados")
        
        styled_df = res_df.style.highlight_max(subset=['R²'], color='lightgreen')\
                                .highlight_min(subset=['MSE', 'RMSE', 'MAE'], color='lightgreen')\
                                .format({'R²': '{:.4f}', 'MSE': '{:.4f}', 'RMSE': '{:.4f}', 'MAE': '{:.4f}'})
        
        st.dataframe(styled_df, use_container_width=True)
        
        mejor_modelo = res_df.loc[res_df['R²'].idxmax(), 'Modelo']
        mejor_r2 = res_df['R²'].max()
        st.success(f"🏆 **Mejor Modelo:** {mejor_modelo} con R² = {mejor_r2:.4f}")
        
        st.markdown("---")
        
        # Gráficos
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            fig_r2 = px.bar(
                res_df, x="Modelo", y="R²", color="Modelo",
                title="Comparación de R²", text="R²"
            )
            fig_r2.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_r2.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_r2, use_container_width=True)
        
        with col_chart2:
            fig_rmse = px.bar(
                res_df, x="Modelo", y="RMSE", color="Modelo",
                title="Comparación de RMSE", text="RMSE"
            )
            fig_rmse.update_traces(texttemplate='%{text:.3f}', textposition='outside')
            fig_rmse.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig_rmse, use_container_width=True)
        
        # Gráfico de radar
        st.markdown("---")
        st.subheader("🎯 Gráfico de Radar Multi-Métrica")
        
        res_radar = res_df.copy()
        res_radar['R²_norm'] = res_radar['R²']
        
        for metric in ['MSE', 'RMSE', 'MAE']:
            min_val = res_radar[metric].min()
            max_val = res_radar[metric].max()
            if max_val - min_val != 0:
                res_radar[f'{metric}_norm'] = 1 - (res_radar[metric] - min_val) / (max_val - min_val)
            else:
                res_radar[f'{metric}_norm'] = 1
        
        fig_radar = go.Figure()
        
        for idx, row in res_radar.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row['R²_norm'], row['MSE_norm'], row['RMSE_norm'], row['MAE_norm']],
                theta=['R²', 'MSE (inv)', 'RMSE (inv)', 'MAE (inv)'],
                fill='toself',
                name=row['Modelo']
            ))
        
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            title="Comparación Multi-Métrica (mayor es mejor)"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Descargar
        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Descargar Resultados",
            data=csv,
            file_name="comparacion_modelos.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    def show_welcome_screen(self):
        """Muestra pantalla de bienvenida"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>👋 Bienvenido al Dashboard de Machine Learning</h2>
            <p style='font-size: 1.2rem;'>Sube un archivo CSV para comenzar</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.info("""
        ### 📋 **Requisitos del Dataset:**
        
        ✅ Formato: **CSV**  
        ✅ Tamaño: **Entre 500 y 1000 filas**  
        ✅ Contenido: **Al menos 2 columnas numéricas**
        """)
    
    def run(self):
        """Método principal que ejecuta la aplicación"""
        if self.load_data():
            if self.validate_data():
                tab1, tab2, tab3, tab4 = st.tabs([
                    "📊 Análisis Exploratorio",
                    "🧹 Limpieza de Datos",
                    "⚙️ Entrenar Modelo",
                    "📈 Comparar Modelos"
                ])
                
                with tab1:
                    self.show_exploratory_analysis()
                
                with tab2:
                    self.show_data_cleaning()
                
                with tab3:
                    self.show_model_training()
                
                with tab4:
                    self.show_model_comparison()
        else:
            self.show_welcome_screen()


# ========================================
# PUNTO DE ENTRADA DE LA APLICACIÓN
# ========================================
if __name__ == "__main__":
    dashboard = MLDashboard()
    dashboard.run()