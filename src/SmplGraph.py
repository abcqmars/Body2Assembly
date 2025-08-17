class UndirectedAcyclicGraph:
    def __init__(self):
        self.adjacency_list = {}
    
    def add_node(self, node):
        if node not in self.adjacency_list:
            self.adjacency_list[node] = []

    def add_edge(self, node1, node2):
        if node1 not in self.adjacency_list:
            self.add_node(node1)
        if node2 not in self.adjacency_list:
            self.add_node(node2)
        if node2 not in self.adjacency_list[node1]:
            self.adjacency_list[node1].append(node2)
        if node1 not in self.adjacency_list[node2]:
            self.adjacency_list[node2].append(node1)

    def has_cycle_util(self, node, visited, parent):
        visited[node] = True
        for neighbor in self.adjacency_list[node]:
            if not visited[neighbor]:
                if self.has_cycle_util(neighbor, visited, node):
                    return True
            elif parent != neighbor:
                return True
        return False

    def is_acyclic(self):
        visited = {node: False for node in self.adjacency_list}
        for node in self.adjacency_list:
            if not visited[node]:
                if self.has_cycle_util(node, visited, None):
                    return False
        return True

    def find_distance(self, start_node, end_node):
        if start_node not in self.adjacency_list or end_node not in self.adjacency_list:
            return -1  # Return -1 if either node is not in the graph

        visited = {node: False for node in self.adjacency_list}
        queue = [(start_node, 0)]  # (current_node, current_distance)

        while queue:
            current_node, current_distance = queue.pop(0)
            if current_node == end_node:
                return current_distance

            visited[current_node] = True
            for neighbor in self.adjacency_list[current_node]:
                if not visited[neighbor]:
                    queue.append((neighbor, current_distance + 1))

        return -1  # Return -1 if no path exists between the nodes

    def get_edges(self):
        edges = []
        for node, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                if (neighbor, node) not in edges and (node, neighbor) not in edges:
                    edges.append((node, neighbor))
        return edges

class SMPLGraph:
    """
    This data strucutre aims to model the smpl body skeleton, while some some parts are ignored, still maintaining the connectivity.
    """
    def __init__(self, valid_joints=None):
        self.full_graph = None
        self.build_full_graph()

        self.reduced_graph = None
        if valid_joints is None:
            self.reduced_graph = self.full_graph
        else:
            self.build_reduced_graph(valid_joints)

    def build_full_graph(self):
        self.full_graph = UndirectedAcyclicGraph()
        for i in range(24):
            self.full_graph.add_node(i)
        self.full_graph.add_edge(0, 1)
        self.full_graph.add_edge(0, 2)
        self.full_graph.add_edge(1, 4)
        self.full_graph.add_edge(2, 5)
        self.full_graph.add_edge(4, 7)
        self.full_graph.add_edge(5, 8)
        self.full_graph.add_edge(7, 10)
        self.full_graph.add_edge(8, 11)
        self.full_graph.add_edge(0, 3)
        self.full_graph.add_edge(3, 6)
        self.full_graph.add_edge(6, 9)
        self.full_graph.add_edge(9, 13)
        self.full_graph.add_edge(9, 14)
        self.full_graph.add_edge(9, 12)
        self.full_graph.add_edge(12, 15)
        self.full_graph.add_edge(13, 16)
        self.full_graph.add_edge(16, 18)
        self.full_graph.add_edge(18, 20)
        self.full_graph.add_edge(20, 22)
        self.full_graph.add_edge(14, 17)
        self.full_graph.add_edge(17, 19)
        self.full_graph.add_edge(19, 21)
        self.full_graph.add_edge(21, 23)
    
    def build_reduced_graph(self, valid_joints):
        """
        Build the skeleton graph with only fewer joints.
        """
        self.reduced_graph = None
        assert all(n>=0 and n<24 for n in valid_joints), f"{valid_joints} exceeds the range of 0-23."
        # Build the distances dict.
        distances = {}
        for i in range(len(valid_joints)):
            for j in range(i + 1, len(valid_joints)):
                dist = self.full_graph.find_distance(valid_joints[i], valid_joints[j])
                if dist != -1:
                    distances[(valid_joints[i], valid_joints[j])] = dist
        
        newedges = set()
        for joint in valid_joints:
            min_distance = float('inf')
            edges_to_add = []
            for (joint1, joint2), distance in distances.items():
                if joint in (joint1, joint2):
                    if distance < min_distance:
                        min_distance = distance
                        edges_to_add = [(joint1, joint2)]
                    elif distance == min_distance:
                        edges_to_add.append((joint1, joint2))

            for edge in edges_to_add:
                newedges.add(edge)

        self.reduced_graph = UndirectedAcyclicGraph()
        for joint in valid_joints:
            self.reduced_graph.add_node(joint)
        for edge in newedges:
            self.reduced_graph.add_edge(*edge)

if __name__ == "__main__":
    # Test Example
    smpl_graph = SMPLGraph(valid_joints=[0, 1, 2, 9, 18, 19])