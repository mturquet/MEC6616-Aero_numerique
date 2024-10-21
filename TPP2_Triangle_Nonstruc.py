import numpy as np
import sympy as sp
import scipy.sparse as sparse
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mesh import Mesh
from meshGenerator import MeshGenerator
from meshConnectivity import MeshConnectivity
from meshPlotter import MeshPlotter
import pyvista as pv
import pyvistaqt as pvQt

class ConvectionDiffusionSolver:
    def __init__(self, x_min, x_max, y_min, y_max,lc):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.lc = lc
        self.rho = 1.0
        self.Cp = 1.0
        
        # Définition des symboles pour les valeurs x et y 
        self.x, self.y = sp.symbols('x y')
        
        # Création des champs u et v et de la solution littérale manufacturée 
        self.u = (2*self.x**2 - self.x**4 - 1) * (self.y - self.y**3)
        self.v = -(2*self.y**2 - self.y**4 - 1) * (self.x - self.x**3)
        self.T0, self.Tx, self.Txy = 400, 50, 100
        self.T_mms = self.T0 + self.Tx * sp.cos(sp.pi * self.x) + self.Txy * sp.sin(sp.pi * self.x * self.y)
        
        # Création du terme source 
        self.S_func_generator = self.compute_terme_source()
        
        # Créations des fonctions python 
        self.u_func = sp.lambdify((self.x, self.y), self.u, 'numpy')
        self.v_func = sp.lambdify((self.x, self.y), self.v, 'numpy')
        self.T_mms_func = sp.lambdify((self.x, self.y), self.T_mms, 'numpy')
        
        self.mesh = None
        self.face_areas = None
        self.cell_volumes = None
        self.normale_face = None
        self.centre_cellule = None

    def compute_terme_source(self):
        k = sp.symbols('k')
        #Calcul des termes de convection 
        convection_x = sp.diff(self.rho * self.u * self.Cp * self.T_mms, self.x)
        convection_y = sp.diff(self.rho * self.v * self.Cp * self.T_mms, self.y)
        # Calcul des termes de diffusion 
        diffusion_x = sp.diff(k * sp.diff(self.T_mms, self.x), self.x)
        diffusion_y = sp.diff(k * sp.diff(self.T_mms, self.y), self.y)
        # Calcul du terme source par méthode MMS
        S = convection_x + convection_y - diffusion_x - diffusion_y
        
        # Création de la fonction source : fonction 
        def S_func_generator(k_val):
            return sp.lambdify((self.x, self.y), S.subs(k, k_val), 'numpy')
        
        return S_func_generator
    
    # Maillage
    def generate_mesh(self, mesh_type='TRI'):
        mesh_generator = MeshGenerator()
        mesh_parameters = {
            'mesh_type': mesh_type,
            'lc': self.lc
        }
        self.mesh = mesh_generator.rectangle([self.x_min, self.x_max, self.y_min, self.y_max], mesh_parameters)
        mesh_connectivity = MeshConnectivity(self.mesh)
        mesh_connectivity.compute_connectivity()
        self.compute_mesh_properties() 
        return self.mesh

    def plot_mesh(self):
        plotter = MeshPlotter()
        plotter.plot_mesh(self.mesh, label_elements=True, label_faces=True)
        self.compute_cross_diffusion_vectors()
    # Cross-diffusion    
    def compute_cross_diffusion_vectors(self):
        cross_diff_vectors = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)
            face_center = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
            if right_cell != -1:  # Internal face
                cell_vector = self.centre_cellule[right_cell] - self.centre_cellule[left_cell]
                cross_diff_vectors[i_face] = face_center - (self.centre_cellule[left_cell] + 0.5 * cell_vector)
            else:  # Boundary face
                cross_diff_vectors[i_face] = face_center - self.centre_cellule[left_cell]
        return cross_diff_vectors

    # Création des paramètres propres au maillage pour les utiliser plus loin            
    def compute_mesh_properties(self):
        self.face_areas = self.compute_longueur_face()
        self.cell_volumes = self.compute_cell_volumes()
        self.normale_face = self.compute_normale_face()
        self.centre_cellule = self.compute_centre_cellule()
        self.cross_diff_vectors = self.compute_cross_diffusion_vectors() 
        
    # Définition de la surface des cellules 
    def compute_longueur_face(self):
        surface_face= np.zeros(self.mesh.get_number_of_faces())
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            surface_face[i_face] = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return surface_face
    
    # Nouvelle fonction de création des volumes 
    def compute_cell_volumes(self):
        cell_volumes = np.zeros(self.mesh.get_number_of_elements())
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            if len(nodes) == 3:  # Triangles
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                cell_volumes[i_elem] = 0.5 * abs((x2 - x1)*(y3 - y1) - (x3 - x1)*(y2 - y1))
            elif len(nodes) == 4:  # Quadrilatères
                x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
                x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
                x3, y3 = self.mesh.get_node_to_xycoord(nodes[2])
                x4, y4 = self.mesh.get_node_to_xycoord(nodes[3])
                cell_volumes[i_elem] = 0.5 * abs((x3 - x1)*(y4 - y2) - (x4 - x2)*(y3 - y1)) # COM : Correction TPP1
        return cell_volumes

    def compute_normale_face(self):
        normale_face = np.zeros((self.mesh.get_number_of_faces(), 2))
        for i_face in range(self.mesh.get_number_of_faces()):
            nodes = self.mesh.get_face_to_nodes(i_face)
            x1, y1 = self.mesh.get_node_to_xycoord(nodes[0])
            x2, y2 = self.mesh.get_node_to_xycoord(nodes[1])
            nx = y2 - y1
            ny = x1 - x2
            norm = np.sqrt(nx**2 + ny**2)
            normale_face[i_face] = [nx/norm, ny/norm]
        return normale_face

    def compute_centre_cellule(self):
        centre_cellule = np.zeros((self.mesh.get_number_of_elements(), 2))
        for i_elem in range(self.mesh.get_number_of_elements()):
            nodes = self.mesh.get_element_to_nodes(i_elem)
            x_coords = np.array([self.mesh.get_node_to_xcoord(node) for node in nodes])
            y_coords = np.array([self.mesh.get_node_to_ycoord(node) for node in nodes])
            centre_cellule[i_elem] = [np.mean(x_coords), np.mean(y_coords)]
        return centre_cellule

# Assemblage matriciel
    def assemble_system(self, k, scheme='centré'):
        n_cells = self.mesh.get_number_of_elements() 
        A = sparse.lil_matrix((n_cells, n_cells))
        b = np.zeros(n_cells)
        
        S_func = self.S_func_generator(k) # Création de S en fonction de k 
        
        for i_face in range(self.mesh.get_number_of_faces()): # Pour toutes les arrêtes
            left_cell, right_cell = self.mesh.get_face_to_elements(i_face)  # Triangle de gauche et de droite 
            
            if right_cell != -1:  #  Arrête non limites
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0) # Récupération du centre de l'arrête
                u_face = self.u_func(x_face, y_face) # Champ u au milieu 
                v_face = self.v_func(x_face, y_face) # Champ v au milieu
                velocity_dot_normal = u_face * self.normale_face[i_face, 0] + v_face * self.normale_face[i_face, 1] # récupération du produit scalaire
                
                # Terme convectif
                if scheme == 'centré':
                    coeff = 0.5 * self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    A[left_cell, right_cell] -= coeff
                    A[right_cell, left_cell] += coeff
                    A[left_cell, left_cell] += coeff
                    A[right_cell, right_cell] -= coeff
                    
                elif scheme == 'upwind':
                    coeff = self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    if velocity_dot_normal > 0:
                        A[right_cell, left_cell] += coeff
                        A[left_cell, left_cell] -= coeff
                    else:
                        A[left_cell, right_cell] -= coeff
                        A[right_cell, right_cell] += coeff
                
                # Terme diffusif avec cross_diff
                
                dx = self.centre_cellule[right_cell] - self.centre_cellule[left_cell]
                distance = np.linalg.norm(dx)
                diff_coeff = k * self.face_areas[i_face] / distance
                A[left_cell, right_cell] -= diff_coeff
                A[right_cell, left_cell] -= diff_coeff
                A[left_cell, left_cell] += diff_coeff
                A[right_cell, right_cell] += diff_coeff
                
                # Correction Cross-diffusion 
                cross_diff_vector = self.cross_diff_vectors[i_face]
                cross_diff_coeff = k * np.dot(cross_diff_vector, self.normale_face[i_face]) * self.face_areas[i_face] / distance**2
                A[left_cell, left_cell] += cross_diff_coeff
                A[right_cell, right_cell] += cross_diff_coeff
                A[left_cell, right_cell] -= cross_diff_coeff
                A[right_cell, left_cell] -= cross_diff_coeff
            
            else:  # Arrête Limite
                x_face, y_face = np.mean([self.mesh.get_node_to_xycoord(node) for node in self.mesh.get_face_to_nodes(i_face)], axis=0)
                T_boundary = self.T_mms_func(x_face, y_face)
                
                # Terme diffusif avec la correction de  cross-diffusion (Dirichlet condition limite)
                dx = np.array([x_face, y_face]) - self.centre_cellule[left_cell]
                distance = np.linalg.norm(dx)
                diff_coeff = k * self.face_areas[i_face] / distance
                A[left_cell, left_cell] += diff_coeff
                b[left_cell] += diff_coeff * T_boundary
                
                # Correction Cross-diffusion pour limite
                cross_diff_vector = self.cross_diff_vectors[i_face]
                cross_diff_coeff = k * np.dot(cross_diff_vector, self.normale_face[i_face]) * self.face_areas[i_face] / distance**2
                A[left_cell, left_cell] += cross_diff_coeff
                b[left_cell] += cross_diff_coeff * T_boundary
                
                #Terme de convection
                u_face = self.u_func(x_face, y_face)
                v_face = self.v_func(x_face, y_face)
                velocity_dot_normal = u_face * self.normale_face[i_face, 0] + v_face * self.normale_face[i_face, 1]
                if velocity_dot_normal < 0:  # Inflow
                    conv_coeff = self.rho * self.Cp * velocity_dot_normal * self.face_areas[i_face]
                    b[left_cell] -= conv_coeff * T_boundary
        
        # Terme source 
        for i_cell in range(n_cells):
            x_cell, y_cell = self.centre_cellule[i_cell]
            b[i_cell] += S_func(x_cell, y_cell) * self.cell_volumes[i_cell]
        
        return A.tocsr(), b
    
    def get_boundary_type(self, x, y):
        # Définis les conditions limites en fonction des différentes coordonnées
        # POur ce problème on a exactement : 
        if np.isclose(x, -1) or np.isclose(x, 1):
            return 'Dirichlet'
        elif np.isclose(y, -1) or np.isclose(y, 1):
            return 'Neumann'
            

    def get_valeur_neumann(self, x, y):
        # Valeur numérique de DT/dn à partir de Tmms(MMS)
        dTdx = self.Tx * (-np.pi * np.sin(np.pi * x)) + self.Txy * y * np.pi * np.cos(np.pi * x * y) # Créer pour la forme si on change 
        dTdy = self.Txy * x * np.pi * np.cos(np.pi * x * y)
        
        if np.isclose(y, -1):
            return -dTdy  # normal is (0, -1)
        elif np.isclose(y, 1):
            return dTdy   # normal is (0, 1)
    
    
    def solve_system(self, A, b):
        return spla.spsolve(A, b)

    def compute_error(self, T_numerical):
         error = np.zeros_like(T_numerical)
         for i_cell in range(self.mesh.get_number_of_elements()):
             x_cell, y_cell = self.centre_cellule[i_cell]
             error[i_cell] = abs(T_numerical[i_cell] - self.T_mms_func(x_cell, y_cell))
         
         L1_error = np.sum(np.abs(error) * self.cell_volumes)/ (self.mesh.get_number_of_elements())
         L2_error = np.sqrt(np.sum(np.abs(error)**2 * self.cell_volumes))/self.mesh.get_number_of_elements()
         Linf_error = np.max(np.abs(error))
         
         return L1_error, L2_error, Linf_error    

    def plot_resultats(self, T_numerical, k, scheme):
        nodes = np.array([self.mesh.get_node_to_xycoord(i) for i in range(self.mesh.get_number_of_nodes())])
        elements = np.array([self.mesh.get_element_to_nodes(i) for i in range(self.mesh.get_number_of_elements())])
        

        triangles = []
        triangle_to_element = []
        for i, element in enumerate(elements):
            if len(element) == 3:
                triangles.append(element)
                triangle_to_element.append(i)
            elif len(element) == 4:
                triangles.append([element[0], element[1], element[2]])
                triangles.append([element[0], element[2], element[3]])
                triangle_to_element.extend([i, i])
        
        triangles = np.array(triangles)
        triangle_to_element = np.array(triangle_to_element)
        

        T_triangles = T_numerical[triangle_to_element]
        
        tri = Triangulation(nodes[:, 0], nodes[:, 1], triangles)

        plt.figure(figsize=(10, 8))
        plt.tripcolor(tri, T_triangles)
        plt.colorbar(label='Température')
        plt.title(f'Distribution de température (Pe={1/k}, {scheme} scheme)')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

    def calculate_convergence_order(self, errors):
       convergence_orders = {scheme: {k: [] for k in errors[scheme]} for scheme in errors}
       
       for scheme in errors:
           for k in errors[scheme]:
               error_data = errors[scheme][k]
               for i in range(0, len(error_data)):
                   h1, e1 = error_data[i-1]
                   h2, e2 = error_data[i]
                   order = np.log(e1/e2) / np.log(h2/h1)
                   convergence_orders[scheme][k].append((h2, order))
       
       return convergence_orders

class ConvectionDiffusionAnalyzer:
    def __init__(self, x_min, x_max, y_min, y_max, lc):
        self.x_min, self.x_max = x_min, x_max
        self.y_min, self.y_max = y_min, y_max
        self.solver = ConvectionDiffusionSolver(x_min, x_max, y_min, y_max,lc)
        self.lc=lc
        self.mesh = None
        self.T_mms_values = None

    def generate_mesh(self, lc):
        self.mesh = self.solver.generate_mesh(lc)
        return self.mesh

    def plot_convergence(self, mesh_sizes, errors,L):
        plt.figure(figsize=(15, 5))
        error_types = ['L1', 'L2', 'Linf']
        for i, error_type in enumerate(error_types):
            plt.subplot(1, 3, i+1)
            for scheme in errors:
                for k, error_values in errors[scheme][error_type].items():
                    plt.loglog(mesh_sizes, error_values, '-o', label=f'{scheme}, Pe={1/k}')
            
            plt.xlabel('Taille de cellule')
            plt.ylabel(f'{error_type} Erreur')
            plt.title(f"{error_type} Convergeance de l'erreur")
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
    def run_convergence_study(self, k_values, schemes, mesh_sizes):
        errors = {scheme: {error_type: {k: [] for k in k_values} for error_type in ['L1', 'L2', 'Linf']} for scheme in schemes}
        
        for lc in mesh_sizes:
            self.solver = ConvectionDiffusionSolver(self.x_min, self.x_max, self.y_min, self.y_max,lc)
            self.generate_mesh(lc) 

            self.solver.compute_mesh_properties() 
            
            for scheme in schemes:
                for k in k_values:
                    A, b = self.solver.assemble_system(k, scheme=scheme)
                    T_numerical = self.solver.solve_system(A, b)
                    L1_error, L2_error, Linf_error = self.solver.compute_error(T_numerical)
                    
                    errors[scheme]['L1'][k].append(L1_error)
                    errors[scheme]['L2'][k].append(L2_error)
                    errors[scheme]['Linf'][k].append(Linf_error)
        
        return errors
    
    def calculate_max_slope(self, mesh_sizes, errors):
        max_slopes = {scheme: {error_type: {} for error_type in ['L1', 'L2', 'Linf']} for scheme in errors}
        
        for scheme in errors:
            for error_type in errors[scheme]:
                for k, error_values in errors[scheme][error_type].items():
                    slopes = []
                    for i in range(1, len(mesh_sizes)):
                        dx = np.log(mesh_sizes[i]) - np.log(mesh_sizes[i-1])
                        dy = np.log(error_values[i]) - np.log(error_values[i-1])
                        slope = abs(dy / dx)
                        slopes.append(slope)
                    max_slopes[scheme][error_type][k] = max(slopes)
        
        return max_slopes
    
    def print_max_slopes(self, max_slopes):
        print("\nOrdre de convergence:")
        for scheme in max_slopes:
            print(f"\nSchéma: {scheme}")
            for error_type in max_slopes[scheme]:
                print(f" Erreur {error_type} :")
                for k, slope in max_slopes[scheme][error_type].items():
                    print(f"   Pe={1/k:.0e}: {slope:.2f}")

    def plot_results(self, T_numerical, k, scheme):
        grid = self.create_pyvista_grid(T_numerical)

        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(grid, show_edges=True, scalars="Température", cmap="RdBu")
        contours = grid.contour(scalars='Température', isosurfaces=10)
        pl.add_mesh(contours, color="white", line_width=2)
        pl.add_text(f'Distribution de température (Pe={1/k}, {scheme} scheme)', font_size=12)
        pl.show()

    def get_temperature_profile(self, grid, start_point, end_point):
        line = grid.sample_over_line(start_point, end_point, resolution=100)
        return line['Distance'], line['Température']

    def create_pyvista_grid(self, T_values):
        plotter = MeshPlotter()  
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)
        cells = elements
        
        cell_types = []
        i = 0
        while i < len(cells):
            num_points = cells[i]
            if num_points == 3:
                cell_types.append(pv.CellType.TRIANGLE)
            elif num_points == 4:
                cell_types.append(pv.CellType.QUAD)
            i += num_points + 1
        
        cell_types = np.array(cell_types)
        
        grid = pv.UnstructuredGrid(cells, cell_types, points)

        if len(T_values) == grid.n_cells:
            grid.cell_data['Température'] = T_values
            grid = grid.cell_data_to_point_data()
        elif len(T_values) == grid.n_points:
            grid.point_data['Température'] = T_values
        
        return grid

    def plot_cross_section_profiles(self, profiles, analytical_profile, title, xlabel):
        plt.figure(figsize=(12, 8))
        
        for k, scheme, (distance, temperature) in profiles:

            plt.plot(distance, temperature, label=f' Pe={1/k}, {scheme} scheme')
        
        plt.plot(analytical_profile[0], analytical_profile[1], 'k--', label='Analytique')
        
        plt.xlabel(xlabel)
        plt.ylabel('Température')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_T_mms(self):
        plotter = MeshPlotter()  
        nodes, elements = plotter.prepare_data_for_pyvista(self.mesh)
        points = np.array(nodes)

        x, y = sp.symbols('x y')
        T0, Tx, Txy = 400, 50, 100
        T_mms = T0 + Tx * sp.cos(sp.pi * x) + Txy * sp.sin(sp.pi * x * y)
        
        T_mms_func = sp.lambdify((x, y), T_mms, 'numpy')
        
        self.T_mms_values = T_mms_func(points[:, 0], points[:, 1])
        
        grid = self.create_pyvista_grid(self.T_mms_values)
        
        pl = pvQt.BackgroundPlotter()
        pl.add_mesh(grid, scalars='Température', show_edges=True, cmap="RdBu", clim=[self.T_mms_values.min(), self.T_mms_values.max()])
        contours = grid.contour(scalars='Température', isosurfaces=10)
        pl.add_mesh(contours, color="white", line_width=2)
        pl.add_text('T_mms Distribution', font_size=12)
        pl.show()

    def run_analysis(self, k_values, schemes, lc ,mesh_sizes):
        self.generate_mesh(lc)
        self.plot_T_mms()
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        z_min, z_max = 0, 0
        L=x_max-x_min
        cross_sections = [
            ("Coupe Verticale à x=0", [0, y_min, z_min], [0, y_max, z_max], "Distance  y"),
            ("Coupe Verticale à x=0.5", [0.5, y_min, z_min], [0.5, y_max, z_max], "Distance  y"),
            ("Coupe Horizontale à y=0", [x_min, 0, z_min], [x_max, 0, z_max], "Distance  x"),
            ("Coupe Horizontale à y=0.5", [x_min, 0.5, z_min], [x_max, 0.5, z_max], "Distance  x")
        ]

        for k in k_values:
            for scheme in schemes:
                A, b = self.solver.assemble_system(k, scheme=scheme)
                T_numerical = self.solver.solve_system(A, b)
                self.plot_results(T_numerical, k, scheme)

        for section_name, start_point, end_point, xlabel in cross_sections:
            profiles = []
            
            for k in k_values:
                for scheme in schemes:
                    A, b = self.solver.assemble_system(k, scheme=scheme)
                    T_numerical = self.solver.solve_system(A, b)
                    grid = self.create_pyvista_grid(T_numerical)
                    distance, temperature = self.get_temperature_profile(grid, start_point, end_point)
                    profiles.append((k, scheme, (distance, temperature)))
            
            grid_analytical = self.create_pyvista_grid(self.T_mms_values)
            analytical_profile = self.get_temperature_profile(grid_analytical, start_point, end_point)
            
            self.plot_cross_section_profiles(profiles, analytical_profile, f"Distribution de température - {section_name}", xlabel)
        errors = self.run_convergence_study(k_values, schemes, mesh_sizes)
        max_slopes = self.calculate_max_slope(mesh_sizes, errors)
        self.plot_convergence(mesh_sizes, errors, L)
        self.print_max_slopes(max_slopes)

      
def main():
    lc=.1
    analyzer = ConvectionDiffusionAnalyzer(-1, 1, -1, 1, lc)

    k_values = [1, 1/100, 1/10000]

    schemes = ['centré', 'upwind']
    
    mesh_sizes = np.array([lc, lc/2, lc/4])
    
    analyzer.run_analysis(k_values, schemes, lc , mesh_sizes)

    
    
if __name__ == "__main__":
    main()