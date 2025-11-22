#Before all please pip install the following:
pip install scikit-learn flask pandas matplotlib numpy openpyxl requests

#Exercise 1
import requests

a, b = 0.5, 0.5
lr = 0.01
h = 1e-5

for i in range(1000):
    e = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b}", 
                           headers={"User-Agent": "MyScript"}).text)
    ea = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a+h}&b={b}", 
                            headers={"User-Agent": "MyScript"}).text)
    eb = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a}&b={b+h}", 
                            headers={"User-Agent": "MyScript"}).text)
    
    grad_a = (ea - e) / h
    grad_b = (eb - e) / h
    
    a_new = a - lr * grad_a
    b_new = b - lr * grad_b
    
    if abs(a_new - a) < 1e-6 and abs(b_new - b) < 1e-6:
        break
    
    a, b = a_new, b_new

print(f"a={a:.3f}, b={b:.3f}, error={e:.3f}")


a2, b2 = 0.7, 0.3

for i in range(1000):
    e2 = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a2}&b={b2}", 
                            headers={"User-Agent": "MyScript"}).text)
    ea2 = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a2+h}&b={b2}", 
                             headers={"User-Agent": "MyScript"}).text)
    eb2 = float(requests.get(f"http://ramcdougal.com/cgi-bin/error_function.py?a={a2}&b={b2+h}", 
                             headers={"User-Agent": "MyScript"}).text)
    
    a2_new = a2 - lr * (ea2 - e2) / h
    b2_new = b2 - lr * (eb2 - e2) / h
    
    if abs(a2_new - a2) < 1e-6 and abs(b2_new - b2) < 1e-6:
        break
    
    a2, b2 = a2_new, b2_new

# Compare with previous results
if e < e2:
    print(f"gm: a={a:.3f}, b={b:.3f}, error={e:.3f}")
    print(f"lm: a={a2:.3f}, b={b2:.3f}, error={e2:.3f}")
else:
    print(f"gm: a={a2:.3f}, b={b2:.3f}, error={e2:.3f}")
    print(f"lm: a={a:.3f}, b={b:.3f}, error={e:.3f}")


#Exercise 2
import numpy as np

def align(seq1, seq2, match=1, gap_penalty=1, mismatch_penalty=1):
    m, n = len(seq1), len(seq2)
    H = np.zeros((m + 1, n + 1), dtype=int)
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            match_score = H[i-1, j-1] + (match if seq1[i-1] == seq2[j-1] else -mismatch_penalty)
            delete = H[i-1, j] - gap_penalty
            insert = H[i, j-1] - gap_penalty
            H[i, j] = max(0, match_score, delete, insert)
    
    # Find maximum score
    max_score = np.max(H)
    max_pos = np.unravel_index(np.argmax(H), H.shape)
    
    # Backtrack
    align1, align2 = '', ''
    i, j = max_pos
    while i > 0 and j > 0 and H[i, j] > 0:
        score = H[i, j]
        diag = H[i-1, j-1]
        up = H[i-1, j]
        left = H[i, j-1]
        
        if score == diag + (match if seq1[i-1] == seq2[j-1] else -mismatch_penalty):
            align1 = seq1[i-1] + align1
            align2 = seq2[j-1] + align2
            i -= 1
            j -= 1
        elif score == up - gap_penalty:
            align1 = seq1[i-1] + align1
            align2 = '-' + align2
            i -= 1
        else:
            align1 = '-' + align1
            align2 = seq2[j-1] + align2
            j -= 1
    
    return align1, align2, max_score

# 1
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac')
print("Test 1 (default parameters):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 2
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', gap_penalty=2)
print("Test 2 (gap_penalty=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 3
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', match=2)
print("Test 3 (match=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")

# 4
seq1, seq2, score = align('tgcatcgagaccctacgtgac', 'actagacctagcatcgac', mismatch_penalty=2)
print("Test 4 (mismatch_penalty=2):")
print(f"seq1 = {seq1}")
print(f"seq2 = {seq2}")
print(f"score = {score}\n")


#Exercise 3
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

data = pd.read_excel('Rice_Cammeo_Osmancik.xlsx')
my_cols = data.columns[:-1]
mean = data[my_cols].mean()
std = data[my_cols].std()
data[my_cols] = (data[my_cols] - mean) / std

pca = PCA(n_components=2)
data_reduced = pca.fit_transform(data[my_cols])
pc0 = data_reduced[:, 0]
pc1 = data_reduced[:, 1]

plt.figure(figsize=(10, 6))
for rice_type in data['Class'].unique():
    mask = data['Class'] == rice_type
    plt.scatter(pc0[mask], pc1[mask], label=rice_type, alpha=0.6)
plt.xlabel('PC0')
plt.ylabel('PC1')
plt.legend()
plt.title('Rice Types in 2D PCA Space')
plt.show()


class QuadTree:
    def __init__(self, points, classes, max_points=10):
        self.points = points
        self.classes = classes
        self.max_points = max_points
        self.children = None
        
        if len(points) > max_points:
            self._subdivide()
    
    def _subdivide(self):
        x_mid = (self.points[:, 0].min() + self.points[:, 0].max()) / 2
        y_mid = (self.points[:, 1].min() + self.points[:, 1].max()) / 2
        
        masks = [
            (self.points[:, 0] <= x_mid) & (self.points[:, 1] <= y_mid),
            (self.points[:, 0] > x_mid) & (self.points[:, 1] <= y_mid),
            (self.points[:, 0] <= x_mid) & (self.points[:, 1] > y_mid),
            (self.points[:, 0] > x_mid) & (self.points[:, 1] > y_mid)
        ]
        
        self.children = []
        for mask in masks:
            if mask.sum() > 0:
                self.children.append(QuadTree(self.points[mask], self.classes[mask], self.max_points))
    
    def contains(self, x, y):
        x_min, x_max = self.points[:, 0].min(), self.points[:, 0].max()
        y_min, y_max = self.points[:, 1].min(), self.points[:, 1].max()
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    def all_points(self):
        if self.children is None:
            return self.points, self.classes
        all_pts = []
        all_cls = []
        for child in self.children:
            pts, cls = child.all_points()
            all_pts.append(pts)
            all_cls.append(cls)
        return np.vstack(all_pts), np.concatenate(all_cls)


def knn_classify(tree, x, y, k):
    def find_containing_tree(tree, x, y):
        if tree.children is None:
            return tree
        for child in tree.children:
            if child.contains(x, y):
                return find_containing_tree(child, x, y)
        return tree
    
    start_tree = find_containing_tree(tree, x, y)
    
    def search_radius(tree, x, y, radius):
        points, classes = tree.all_points()
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        
        if (x - radius > x_max or x + radius < x_min or 
            y - radius > y_max or y + radius < y_min):
            return [], []
        
        distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
        within = distances <= radius
        return points[within], classes[within]
    
    def get_all_within_radius(tree, x, y, radius):
        if tree.children is None:
            return search_radius(tree, x, y, radius)
        
        all_pts = []
        all_cls = []
        pts, cls = search_radius(tree, x, y, radius)
        if len(pts) > 0:
            all_pts.append(pts)
            all_cls.append(cls)
        
        for child in tree.children:
            pts, cls = get_all_within_radius(child, x, y, radius)
            if len(pts) > 0:
                all_pts.append(pts)
                all_cls.append(cls)
        
        if len(all_pts) == 0:
            return np.array([]).reshape(0, 2), np.array([])
        return np.vstack(all_pts), np.concatenate(all_cls)
    
    current_tree = start_tree
    while True:
        points, classes = current_tree.all_points()
        if len(points) >= k:
            break
        if hasattr(current_tree, '_parent'):
            current_tree = current_tree._parent
        else:
            points, classes = tree.all_points()
            break
    
    distances = np.sqrt((points[:, 0] - x)**2 + (points[:, 1] - y)**2)
    min_i = np.argpartition(distances, min(k, len(distances)-1))[:k]
    nearest_classes = classes[min_i]
    
    return pd.Series(nearest_classes).mode()[0]


X = np.column_stack([pc0, pc1])
y = data['Class'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
tree = QuadTree(X_train, y_train)


predictions_k1 = [knn_classify(tree, x, y, 1) for x, y in X_test]
cm_k1 = confusion_matrix(y_test, predictions_k1)
print(cm_k1)

predictions_k5 = [knn_classify(tree, x, y, 5) for x, y in X_test]
cm_k5 = confusion_matrix(y_test, predictions_k5)
print(cm_k5)


#Exercise 4
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


data = pd.read_csv('eeg-eye-state.csv')
data = data.drop(columns=["eyeDetection"])
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)


pca = PCA(n_components=14)
data_pca = pca.fit_transform(data_standardized)

plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Plot before zoomed in')
plt.show()


plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1])
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('PCA Plot after zoomed in')
plt.xlim(-2, 4)
plt.ylim(-1.5, 1.5)
plt.show()


kmeans = KMeans(n_clusters=7, init='random', random_state=42)
clusters = kmeans.fit_predict(data_pca)

centers_pca = kmeans.cluster_centers_

plt.figure(figsize=(10, 6))
for i in range(7):
    mask = clusters == i
    plt.scatter(data_pca[mask, 0], data_pca[mask, 1], label=f'Cluster {i}')
plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='x', label='Cluster Centers')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('K-Means Clustering')
plt.xlim(-2, 4)
plt.ylim(-1.5, 1.5)
plt.legend()
plt.show()


#Exercise 6
import urllib.request
import zipfile

urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/00296/dataset_diabetes.zip", "dataset_diabetes.zip")
zipfile.ZipFile("dataset_diabetes.zip").extractall(".")


open("templates/index.html", "w").write("""<html>
<body>
<*STATIC IMAGE*, insert an image at the page>
<img src="/static/icon.png" width="100">
<br>
Enter an age range to find average hospital stay:<br>
<form action="/analyze" method="POST">
<input type="text" name="agerange" placeholder="e.g., [50-60)">
<br>
<input type="submit" value="Analyze">
</form>
</body>
</html>""")


open("templates/analyze.html", "w").write("""<html>
<body>
You entered: {{ agerange }}<br><br>
{% if error %}
<*ERROR HANDLING* give red color>
<p style="color:red">{{ error }}</p>
{% else %}
<*MULTIPLE ANALYSES* not only average but also range and mode>
<p>Number of patients: {{ n }}</p>
<p>Average stay: {{ avg }} days</p>
<p>Range: {{ min }} - {{ max }} days</p>
<p>Mode: {{ mode }} days</p>
<*DYNAMIC IMAGE* plot different image base on imput age>
<img src="/static/chart.png" width="600">
{% endif %}
<br><br>
<a href="/">Back</a>
</body>
</html>""")



import os
os.makedirs("templates", exist_ok=True)
os.makedirs("static", exist_ok=True)
# then I put in an image named 'icon.png" inside the folder 'static'


open("server.py", "w").write("""from flask import Flask, render_template, request
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    agerange = request.form["agerange"]
    df = pd.read_csv("dataset_diabetes/diabetic_data.csv")
    filtered = df[df["age"] == agerange]
    
    #ERROR HANDLING (tell no patient in the given range)
    if len(filtered) == 0:
        return render_template("analyze.html", agerange=agerange, error="No patients found in that age range. Try: [50-60), [70-80), [30-40)")
    
    #MULTIPLE ANALYSES (range and mode)
    avg = filtered["time_in_hospital"].mean()
    min_val = filtered["time_in_hospital"].min()
    max_val = filtered["time_in_hospital"].max()
    mode_val = filtered["time_in_hospital"].mode()[0]
    
    #GENERATE DYNAMIC IMAGE (based on given age range)
    plt.figure(figsize=(8, 5))
    filtered["time_in_hospital"].hist(bins=15, edgecolor='black')
    plt.title(f'Hospital Stay Distribution for Age {agerange}')
    plt.xlabel('Days in Hospital')
    plt.ylabel('Number of Patients')
    plt.savefig('static/chart.png')
    plt.close()
    
    return render_template("analyze.html", agerange=agerange, n=len(filtered),
                         avg=f"{avg:.2f}", min=min_val, max=max_val, mode=mode_val)

if __name__ == "__main__":
    app.run()
""")
