import networkx as nx
import geopandas as gpd
import os
import time

def convert_attributes_to_string(G):
    for u, v, attributes in G.edges(data=True):
        for key, value in attributes.items():
            if not isinstance(value, (int, float, str, bool)):
                G.edges[u, v][key] = str(value)

    for node, attributes in G.nodes(data=True):
        for key, value in attributes.items():
            if not isinstance(value, (int, float, str, bool)):
                G.nodes[node][key] = str(value)

def round_coordinates(coords, digits=3):
    return tuple(round(coord, digits) for coord in coords)

def gdf2graph(gdf):
    G = nx.Graph()
    for _, row in gdf.iterrows():
        start_point = round_coordinates(row['geometry'].coords[0])
        end_point = round_coordinates(row['geometry'].coords[-1])
        
        # 除去几何属性，将所有其他属性加入到图的边
        edge_attributes = {key: value for key, value in row.items() if key != 'geometry'}
        G.add_edge(start_point, end_point, **edge_attributes)
    
    return G

def nodegraph2edgegraph(G):
    H = nx.Graph()

    # 创建以道路ID为节点的新图
    for u, v, data in G.edges(data=True):
        road_id = data['ID']
        if not H.has_node(road_id):
            H.add_node(road_id, **data)  # 将道路的所有属性复制到节点

    # 添加边：如果两条道路共享一个节点，则它们之间添加一条边
    for node in G.nodes():
        connected_edges = list(G.edges(node, data=True))
        for i in range(len(connected_edges) - 1):
            for j in range(i + 1, len(connected_edges)):
                edge_i_id = connected_edges[i][2]['ID']
                edge_j_id = connected_edges[j][2]['ID']
                if not H.has_edge(edge_i_id, edge_j_id):
                    # 可以添加更多的属性到这些边，例如计算两条道路的某种相似度或距离
                    H.add_edge(edge_i_id, edge_j_id)

    return H

if __name__ == '__main__':
    t_start = time.time()
    roadPath = 'data\input\pinns-gnn\london_small_single.geojson'
    output_dir = 'data/output/pinns-gnn'

    # 读取GeoJSON文件
    gdf = gpd.read_file(roadPath)
    # 构建图
    G = gdf2graph(gdf)
    # 将道路转换为节点，它们的连接为边
    H = nodegraph2edgegraph(G)
    # 转换属性为字符串（如果需要）
    convert_attributes_to_string(H)
    # 存储图形结构
    nx.write_graphml(H, os.path.join(output_dir, 'network_edges_as_nodes.graphml'))
    print('Time cost: %.2f s' % (time.time() - t_start))

