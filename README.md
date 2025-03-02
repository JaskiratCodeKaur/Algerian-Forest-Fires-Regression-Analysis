<h1 align="center">🔥 Algerian Forest Fires Regression Analysis</h1>
<h3 align="center">Comparing Regression Models for Fire Likelihood Prediction</h3>

<div align="center">
    <img src="https://img.shields.io/badge/Python-3.8+-3776AB?logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Scikit_Learn-FF6F00?logo=scikitlearn&logoColor=white">
    <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white">
    <img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white">
    <img src="https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white">
</div>

<h2>📖 Overview</h2>
<p>This project compares various regression techniques to predict the likelihood of forest fires in Algeria using meteorological and environmental data from the UCI Machine Learning Repository.</p>

<h2>🛠️ Tech Stack</h2>
<table>
    <tr>
        <th>Technology</th>
        <th>Description</th>
        <th>Badge</th>
    </tr>
    <tr>
        <td>Python</td>
        <td>Core programming language for data analysis and modeling</td>
        <td><img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white"></td>
    </tr>
    <tr>
        <td>Scikit-Learn</td>
        <td>Machine learning library for regression model implementation</td>
        <td><img src="https://img.shields.io/badge/Scikit_Learn-FF6F00?logo=scikitlearn&logoColor=white"></td>
    </tr>
    <tr>
        <td>Pandas</td>
        <td>Data manipulation and preprocessing toolkit</td>
        <td><img src="https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white"></td>
    </tr>
    <tr>
        <td>NumPy</td>
        <td>Numerical computing and array processing</td>
        <td><img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white"></td>
    </tr>
    <tr>
        <td>Matplotlib</td>
        <td>Data visualization and results plotting</td>
        <td><img src="https://img.shields.io/badge/Matplotlib-11557C?logo=matplotlib&logoColor=white"></td>
    </tr>
</table>

<h2>📊 Dataset Description</h2>
<ul>
    <li><strong>Dataset Name:</strong> Algerian Forest Fires Dataset</li>
    <li><strong>Source:</strong> <a href="https://archive.ics.uci.edu/ml/datasets/Algerian+Forest+Fires+Dataset">UCI Machine Learning Repository</a></li>
    <li><strong>Features:</strong>
        <ul>
            <li>🌡️ Meteorological: Temperature, Humidity, Wind Speed, Rain</li>
            <li>🔥 FWI Components: FFMC, DMC, DC, ISI, BUI</li>
            <li>📅 Temporal: Day, Month</li>
        </ul>
    </li>
    <li><strong>Target:</strong> Fire likelihood (continuous variable)</li>
</ul>

<h2>🧠 Algorithms Tested</h2>
<ul>
    <li>📈 Linear Regression</li>
    <li>📉 Polynomial Regression (Best: degree=2)</li>
    <li>📍 K-NN Regression (Best: k=3, Manhattan distance)</li>
    <li>🌳 Decision Tree (Best: unlimited depth)</li>
    <li>🔄 SVR (Best: RBF kernel)</li>
</ul>

<h2>📈 Results</h2>
<table>
    <tr>
        <th>Model</th>
        <th>Mean MSE</th>
        <th>Min MSE</th>
        <th>Max MSE</th>
    </tr>
    <tr>
        <td>Linear Regression</td>
        <td>0.0849</td>
        <td>0.0572</td>
        <td>0.1191</td>
    </tr>
    <tr>
        <td>Polynomial Regression</td>
        <td>0.4995</td>
        <td>0.1268</td>
        <td>1.0032</td>
    </tr>
    <tr>
        <td>K-NN Regression</td>
        <td>0.0380</td>
        <td>0.0088</td>
        <td>0.0546</td>
    </tr>
    <tr>
        <td>Decision Tree</td>
        <td>0.0143</td>
        <td>0.0028</td>
        <td>0.0256</td>
    </tr>
    <tr>
        <td>SVR</td>
        <td>0.0557</td>
        <td>0.0284</td>
        <td>0.0954</td>
    </tr>
</table>

<h2>🚀 Quick Start</h2>
<pre><code>git clone https://github.com/JaskiratCodeKaur/RegressionAnalysis.git
cd algerian-forest-fires-regression
python main.py</code></pre>

<h2>📚 References</h2>
<ul>
    <li>Dataset: <a href="https://doi.org/10.24432/C5KW4N">Algerian Forest Fires Dataset</a></li>
    <li>Scikit-Learn: <a href="https://scikit-learn.org/">Official Documentation</a></li>
</ul>

<div align="center" style="margin-top: 40px;">
    <img src="https://img.shields.io/badge/Made_with-Python-3776AB?logo=python&logoColor=white">
    <img src="https://img.shields.io/badge/Open_Source-❤️-FF6F00">
</div>
