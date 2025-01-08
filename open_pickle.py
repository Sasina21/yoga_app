import pickle
import networkx as nx
import matplotlib.pyplot as plt
import random

# โหลดกราฟจากไฟล์ pickle
pickle_path = "output_graphs.pkl"
with open(pickle_path, 'rb') as f:
    graphs = pickle.load(f)

# กรองกราฟที่มี classification = 0
graphs_class_0 = [G for G in graphs if G.graph.get('classification') == 4]

# ตรวจสอบว่ามีกราฟที่ตรงกับเงื่อนไขหรือไม่
if not graphs_class_0:
    print("ไม่มีกราฟที่มี classification = 0")
else:
    # เลือกกราฟแบบสุ่ม
    random_graph = random.choice(graphs_class_0)

    # แสดงข้อมูลกราฟ
    print(f"กราฟที่เลือกแบบสุ่ม: จำนวนโหนด {random_graph.number_of_nodes()}, จำนวนขอบ {random_graph.number_of_edges()}")

    # กำหนดตำแหน่งโหนด (x, y) และปรับแกน y
    pos = {node: (data['x'], -data['y']) for node, data in random_graph.nodes(data=True)}

    node_colors = ['red' if data['label'] == 1 else 'blue' for node, data in random_graph.nodes(data=True)]

    # ตรวจสอบ attribute 'label' ของโหนดทั้งหมด
    for node, data in random_graph.nodes(data=True):
        print(f"โหนด {node} มีข้อมูล: {data}")

    # แสดงผลกราฟ
    plt.figure(figsize=(8, 6))
    nx.draw(random_graph, pos, with_labels=True, node_size=500, font_size=10)
    plt.title("กราฟสุ่มจาก classification = 0 (2D Visualization)")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis (Flipped)")
    plt.show()
