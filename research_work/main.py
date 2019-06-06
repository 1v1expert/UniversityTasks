import networkx as nx
import matplotlib.pyplot as plt


def Symmetric_Networks():
	G_symmetric = nx.Graph()
	G_symmetric.add_edge('Amitabh Bachchan', 'Abhishek Bachchan')
	G_symmetric.add_edge('Amitabh Bachchan', 'Aamir Khan')
	G_symmetric.add_edge('Amitabh Bachchan', 'Akshay Kumar')
	G_symmetric.add_edge('Amitabh Bachchan', 'Dev Anand')
	G_symmetric.add_edge('Abhishek Bachchan', 'Aamir Khan')
	G_symmetric.add_edge('Abhishek Bachchan', 'Akshay Kumar')
	G_symmetric.add_edge('Abhishek Bachchan', 'Dev Anand')
	G_symmetric.add_edge('Dev Anand', 'Aamir Khan')
	nx.draw_networkx(G_symmetric)
	plt.show()


def Asymmetric_Networks():
	G_asymmetric = nx.DiGraph()
	G_asymmetric.add_edge('A', 'B')
	G_asymmetric.add_edge('A', 'D')
	G_asymmetric.add_edge('C', 'A')
	G_asymmetric.add_edge('D', 'E')
	nx.spring_layout(G_asymmetric)
	nx.draw_networkx(G_asymmetric)
	

def Weighted_Networks():
	G_weighted = nx.Graph()
	G_weighted.add_edge('Amitabh Bachchan', 'Abhishek Bachchan', weight=25)
	G_weighted.add_edge('Amitabh Bachchan', 'Aaamir Khan', weight=8)
	G_weighted.add_edge('Amitabh Bachchan', 'Akshay Kumar', weight=11)
	G_weighted.add_edge('Amitabh Bachchan', 'Dev Anand', weight=1)
	G_weighted.add_edge('Abhishek Bachchan', 'Aaamir Khan', weight=4)
	G_weighted.add_edge('Abhishek Bachchan', 'Akshay Kumar', weight=7)
	G_weighted.add_edge('Abhishek Bachchan', 'Dev Anand', weight=1)
	G_weighted.add_edge('Dev Anand', 'Aaamir Khan', weight=1)
	nx.draw_networkx(G_weighted)


def visualize_facebook_network():
	G_fb = nx.read_edgelist("facebook_combined.txt", create_using=nx.Graph(), nodetype=int)
	print(nx.info(G_fb))

	pos = nx.spring_layout(G_fb)
	betCent = nx.betweenness_centrality(G_fb, normalized=True, endpoints=True)
	node_color = [20000.0 * G_fb.degree(v) for v in G_fb]
	node_size = [v * 10000 for v in betCent.values()]
	plt.figure(figsize=(20, 20))
	nx.draw_networkx(G_fb, pos=pos, with_labels=False, node_color=node_color, node_size=node_size)
	plt.axis('off')
	plt.show()
	
	
if __name__ == '__main__':
	pass