
"""
TPP1- Maillage quadrilatère et triangles (Maillage Quadrilatère)
MEC6616 - Aérodynamique numérique
Date de création: 2024 - 09 - 23
Auteurs: Nicolas Tran et Mathis Turquet
"""
class MaillageQuad():
    def __init__(self):
        import numpy as np
        import pyvista as pv
        import pyvistaqt as pvQt
        from meshGenerator import MeshGenerator
        from meshConnectivity import MeshConnectivity
        from meshPlotter import MeshPlotter
        import matplotlib.pyplot as plt


        # %% Paramètres
        L = 0.02  # Longueur (m)
        x_0 = 0  # Position initiale
        x_f = L  # Position finale
        T_A = 100  # Température à x = A (°C)
        T_B = 200  # Température à x = B (°C)
        k = 0.5  # Conductivité thermique (W/m·K)
        q = 1e6  # Source de chaleur (W/m³)

        Nx = 10 # Nombre de divisions en x
        Ny = 10  # Nombre de divisions en y
        lc = L / 5  # Longueur caractéristique
        dx = L / Nx

        mesher = MeshGenerator()
        plotter = MeshPlotter()
        mesh_parameters1 = {'mesh_type': 'QUAD',
                            'Nx': 10,
                            'Ny': 10
                            }

        mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)

        conec = MeshConnectivity(mesh_obj1)
        conec.compute_connectivity()

        # %% Fonctions utilitaires

        def normal(xa, ya, xb, yb):
            """
            Retourne les composantes a et b du vecteur normal unitaire à une arête.

            Parameters
            ----------
            xa, ya : float
                Coordonnées du point A.
            xb, yb : float
                Coordonnées du point B.

            Returns
            -------
            a, b : float
                Composantes du vecteur normal unitaire.
            """
            dx = xb - xa
            dy = yb - ya
            Delta_A = np.sqrt(dx ** 2 + dy ** 2)

            a = dy / Delta_A
            b = -dx / Delta_A

            return a, b

        def centre_element_2D(mesh_obj, i_element):
            """
            Calcule le centre géométrique d'un élément en 2D.

            Parameters
            ----------
            mesh_obj : object
                Objet maillage.
            i_element : int
                Index de l'élément.

            Returns
            -------
            xmoy, ymoy : float
                Coordonnées du centre géométrique de l'élément.
            """
            start = mesh_obj.get_element_to_nodes_start(i_element)
            fin = mesh_obj.get_element_to_nodes_start(i_element + 1)
            noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]

            xmoy = 0
            ymoy = 0
            for i in range(len(noeuds_i_elements)):
                x, y = mesh_obj.get_node_to_xycoord(noeuds_i_elements[i])
                xmoy += x
                ymoy += y

            xmoy /= len(noeuds_i_elements)
            ymoy /= len(noeuds_i_elements)

            return xmoy, ymoy

        def surface_cellule(noeuds, mesh_obj):
            """
            Calcule la surface d'une cellule à partir de ses nœuds.

            Parameters
            ----------
            noeuds : list
                Liste des nœuds de la cellule.
            mesh_obj : object
                Objet maillage.

            Returns
            -------
            surface : float
                Surface de la cellule.
            """
            n = len(noeuds)
            surface = 0
            points = []

            for i in range(n):
                x, y = mesh_obj.get_node_to_xycoord(noeuds[i])
                points.append((x, y))

            # Calcul de la surface (Formule du polygone - méthode du "shoelace")
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]  # Boucle circulaire sur les points
                surface += x1 * y2 - x2 * y1

            return abs(surface) / 2

        def surface_moyenne(sol_num, mesh_obj):
            """
            Calcule la moyenne quadratique des surfaces des cellules.

            Parameters
            ----------
            sol_num : array
                Solution numérique.
            mesh_obj : object
                Objet maillage.

            Returns
            -------
            moyenne : float
                Moyenne quadratique des surfaces.
            """
            moyenne = 0

            for i in range(len(sol_num)):
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]

                taille_cellule = surface_cellule(noeuds_i_elements, mesh_obj)
                moyenne += taille_cellule ** 2

            moyenne = np.sqrt(moyenne / len(sol_num))

            return moyenne

        def diffusion2D(mesh_obj, bcdata, gradient, gamma, S, phi, matrice):

            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]

            # Création des coordonnées des éléments

            coordonnees_elements_x = []
            coordonnees_elements_y = []

            # Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements

            A = np.zeros((number_of_elements, number_of_elements))
            B = np.zeros((number_of_elements, 1))
            t = np.zeros(number_of_elements)
            Taille = surface_moyenne(t, mesh_obj)
            B += S * Taille
            for i in range(number_of_elements):
    
                # Création du centre gémoétrique des différentes formes

                x_e, y_e = centre_element_2D(mesh_obj, i)

                # Récupération dans deux listes distinctes

                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)

            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                nodes = mesh_obj.get_face_to_nodes(i)
                Tg, Td = neighbours_elements

                if neighbours_elements[1] != -1:

                    neighbours_elements = mesh_obj.get_face_to_elements(i)
                    nodes = mesh_obj.get_face_to_nodes(i)
                    Tg, Td = neighbours_elements

                    # CRéation des points de la l'arrête

                    xa, ya = mesh_obj.get_node_to_xycoord(nodes[0])
                    xb, yb = mesh_obj.get_node_to_xycoord(nodes[1])

                    delta_x = xb - xa
                    delta_y = yb - ya

                    # Création des points d'arrêtes
                    element_A = neighbours_elements[0]
                    element_p = neighbours_elements[1]

                    xA = coordonnees_elements_x[element_A]
                    xp = coordonnees_elements_x[element_p]
                    yA = coordonnees_elements_y[element_A]
                    yp = coordonnees_elements_y[element_p]

                    # Création des paramètres
                    delta_ksi = ((xA - xp) ** 2 + (yA - yp) ** 2) ** (1 / 2)

                    delta_Ai = ((xb - xa) ** 2 + (yb - ya) ** 2) ** (1 / 2)

                    delta_eta = delta_Ai

                    nix, niy = normal(xa, ya, xb, yb)

                    e_ksi = [(xA - xp) / delta_ksi, (yA - yp) / delta_ksi]

                    e_eta = [(xb - xa) / delta_Ai, (ya - yb) / delta_Ai]

                    grad_b = gradient[Tg]
                    grad_a = gradient[Td]
                    # Construction des calculs

                    PNKSI = (delta_y * (xA - xp) / (delta_Ai * delta_ksi)) - (
                                delta_x * (yA - yp) / (delta_Ai * delta_ksi))

                    PKSIETA = ((xA - xp) * (xa - xb) / (delta_ksi * delta_eta)) + (
                                (yA - yp) * (ya - yb) / (delta_ksi * delta_eta))

                    Di = (1 / PNKSI) * -gamma * delta_Ai / delta_ksi

                    if matrice == 0:
                        # Uniquement si cela est connu

                        Sd_cross_i = -gamma * (PKSIETA / PNKSI) * (
                                    ((grad_a[0] + grad_b[0]) / 2) * (xb - xa) / delta_Ai) * (
                                                 ((grad_a[1] + grad_b[1]) / 2) * (yb - ya) / delta_Ai) * delta_Ai

                    if matrice == 1:
                        Sd_cross_i = -gamma * (PKSIETA / PNKSI) * (
                                    ((grad_a[0] + grad_b[0]) / 2) * (xb - xa) / delta_Ai) * (
                                                 ((grad_a[1] + grad_b[1]) / 2) * (yb - ya) / delta_Ai) * delta_Ai

                    t = np.zeros(number_of_elements)
                    Taille = surface_moyenne(t, mesh_obj)
                    # Implémentation des différents paramètres sur A et B

                    A[Tg][Tg] = A[Tg][Tg] + Di

                    A[Td][Td] = A[Td][Td] + Di

                    A[Tg][Td] = - Di + A[Tg][Td]

                    A[Td][Tg] = -Di + A[Td][Tg]

                    B[Tg] = B[Tg] + Sd_cross_i

                    B[Td] = B[Td] - Sd_cross_i

                if neighbours_elements[1] == -1:
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers[tag]

                    if bc_type == 'NEUMANN':

                        neighbours_elements = mesh_obj.get_face_to_elements(i)
                        nodes = mesh_obj.get_face_to_nodes(i)
                        Tg, Td = neighbours_elements

                        # CRéation des points de la l'arrête

                        xa, ya = mesh_obj.get_node_to_xycoord(nodes[0])
                        xb, yb = mesh_obj.get_node_to_xycoord(nodes[1])

                        delta_x = xb - xa
                        delta_y = yb - ya

                        # Création des points d'arrêtes
                        element_A = neighbours_elements[0]
                        element_p = neighbours_elements[1]

                        xA = coordonnees_elements_x[element_A]
                        xp = coordonnees_elements_x[element_p]
                        yA = coordonnees_elements_y[element_A]
                        yp = coordonnees_elements_y[element_p]

                        # Création des paramètres
                        delta_ksi = ((xA - xp) ** 2 + (yA - yp) ** 2) ** (1 / 2)

                        delta_Ai = ((xb - xa) ** 2 + (yb - ya) ** 2) ** (1 / 2)

                        delta_eta = delta_Ai

                        nix, niy = normal(xa, ya, xb, yb)

                        e_ksi = [(xA - xp) / delta_ksi, (yA - yp) / delta_ksi]

                        e_eta = [(xb - xa) / delta_Ai, (ya - yb) / delta_Ai]

                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]

                        # Construction des calculs

                        PNKSI = (delta_y * (xA - xp) / (delta_Ai * delta_ksi)) - (
                                    delta_x * (yA - yp) / (delta_Ai * delta_ksi))

                        PKSIETA = ((xA - xp) * (xa - xb) / (delta_ksi * delta_eta)) + (
                                    (yA - yp) * (ya - yb) / (delta_ksi * delta_eta))

                        Di = (1 / PNKSI) * -gamma * delta_Ai / delta_ksi

                        if matrice == 0:
                            # Uniquement si cela  n'est pas connu

                            Sd_cross_i = -gamma * (PKSIETA / PNKSI) * (
                                        ((grad_a[0] + grad_b[0]) / 2) * (xb - xa) / delta_Ai) * (
                                                     ((grad_a[1] + grad_b[1]) / 2) * (yb - ya) / delta_Ai) * delta_Ai

                        if matrice == 1:
                            Sd_cross_i = -gamma * (PKSIETA / PNKSI) * (
                                        ((grad_a[0] + grad_b[0]) / 2) * (xb - xa) / delta_Ai) * (
                                                     ((grad_a[1] + grad_b[1]) / 2) * (yb - ya) / delta_Ai) * delta_Ai
                            

                        valeur = bc_number

                        B[Tg] = B[Tg] + valeur*gamma*delta_Ai

                    if bc_type == 'DIRICHLET':
                        # Création des paramètres

                        element_A = neighbours_elements[0]  # Inutile
                        element_p = neighbours_elements[1]  # Inutile
                        xa, ya = mesh_obj.get_node_to_xycoord(nodes[0])  # Inutile
                        xb, yb = mesh_obj.get_node_to_xycoord(nodes[1])  # Inutile

                        xA = coordonnees_elements_x[element_A]
                        yA = coordonnees_elements_y[element_A]

                        xp = (xa + xb) / 2
                        yp = (ya + yb) / 2

                        delta_x = xb - xa
                        delta_y = yb - ya
                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]

                        valeur = bc_number

                        delta_ksi = ((xA - xp) ** 2 + (yA - yp) ** 2) ** (1 / 2)  # OK

                        delta_Ai = ((xb - xa) ** 2 + (yb - ya) ** 2) ** (1 / 2)  # OK

                        delta_eta = delta_Ai  # OK

                        nix, niy = normal(xa, ya, xb, yb)  # OK

                        e_ksi = [(xA - xp) / delta_ksi, (yA - yp) / delta_ksi]

                        e_eta = [(xb - xa) / delta_Ai, (ya - yb) / delta_Ai]

                        # Construction des calculs

                        PNKSI = (delta_y * (xA - xp) / (delta_Ai * delta_ksi)) - (
                                    delta_x * (yA - yp) / (delta_Ai * delta_ksi))

                        PKSIETA = ((xA - xp) * (xa - xb) / (delta_ksi * delta_eta)) + (
                                    (yA - yp) * (ya - yb) / (delta_ksi * delta_eta))

                        Di = (1 / PNKSI) * -gamma * delta_Ai / delta_ksi

                        Sd_cross_i = -gamma * (PKSIETA / PNKSI) * delta_Ai * (
                                    ((grad_a[0] + grad_b[0]) / 2) * (xb - xa) / delta_Ai) * (
                                                 ((grad_a[1] + grad_b[1]) / 2) * (yb - ya) / delta_Ai)

                        A[Tg][Tg] = A[Tg][Tg] + Di
                        B[Tg] = B[Tg] + Di * valeur + Sd_cross_i

                    if (bc_type == 'LIBRE'):
                        B[Tg, 0] = B[Tg, 0]
                        B[Tg, 1] = B[Tg, 1]
            solution = np.linalg.solve(A, B)

            return solution, A, B

        def phi(x, y):
            return x

        def least_square(mesh_obj, bcdata, phi):

            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]

            # Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements

            # Création des coordonnées des éléments

            coordonnees_elements_x = []
            coordonnees_elements_y = []

            voisins = []

            f = mesh_obj.number_of_elements
            a = mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            bcdata = mesh_obj.get_boundary_faces_to_tag()

            # Création de la matrice de calcul ATA

            ATA = np.zeros((f, 2, 2))

            # Création de la matrice B

            B = np.zeros((f, 2))

            for i in range(number_of_elements):
                # Récupération des différents noeuds autour d'un éléments

                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]

                # Création du centre gémoétrique des différentes formes

                x_e, y_e = centre_element_2D(mesh_obj, i)

                # Récupération dans deux listes distinctes

                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)

                # Parcours de l'ensemble des arrêtes pour la création des différentes matrices

            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                # Récupération des éléments voisins
                voisins.append(neighbours_elements)

                Tg, Td = neighbours_elements

                if neighbours_elements[1] == -1:  # pour les faces frontières

                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers[tag]

                    if (bc_type == 'DIRICHLET'):
                        # DIRICHLET
                        # Création du centre de l'arrête
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)

                        # Milieu

                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])

                        x_milieu = (xa + xb) / 2
                        y_milieu = (yb + ya) / 2

                        # Dx et Dy
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        dx = (xtg - x_milieu)
                        dy = (ytg - y_milieu)

                        # Pour les arêtes internes

                        ALS = np.zeros((2, 2))

                        # Création des différents paramèetres de la matrice 2x2

                        ALS[0, 0] = dx * dx
                        ALS[1, 0] = dx * dy
                        ALS[0, 1] = dy * dx
                        ALS[1, 1] = dy * dy
                        # Remplissage

                        ATA[Tg] += ALS

                        Phi_A = phi(x_milieu, y_milieu)
                        Phi_tg = phi(xtg, ytg)  # Modifié pour la valeur précédente

                        B[Tg, 0] = B[Tg, 0] + (x_milieu - xtg) * (Phi_A - Phi_tg)
                        B[Tg, 1] = B[Tg, 1] + (y_milieu - ytg) * (Phi_A - Phi_tg)

                    if (bc_type == 'NEUMANN'):
                        # Neumann
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)

                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])

                        nx, ny = normal(xa, ya, xb, yb)

                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        x_milieu = (mesh_obj.get_node_to_xcoord(noeuds_faces[1]) + mesh_obj.get_node_to_xcoord(
                            noeuds_faces[0])) / 2
                        y_milieu = (mesh_obj.get_node_to_ycoord(noeuds_faces[1]) + mesh_obj.get_node_to_ycoord(
                            noeuds_faces[0])) / 2

                        dx1 = ((x_milieu - xtg))
                        dy1 = ((y_milieu - ytg))

                        dx = (dx1 * nx + dy1 * ny) * nx
                        dy = (dx1 * nx + dy1 * ny) * ny

                        # Matric intermédiaire
                        ALS = np.zeros((2, 2))

                        # Création des différents paramètres de la matrice 2x2

                        ALS[0, 0] = dx * dx
                        ALS[1, 0] = ALS[0, 1] = dy * dx
                        ALS[1, 1] = dy * dy

                        # Remplissage
                        ATA[Tg] += ALS

                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        dx = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * nx
                        dy = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * ny

                        Phi_A = bc_number  # phi(x_milieu,y_milieu) # Phi milieu
                        Phi_N = bc_number

                        delta_phi = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * Phi_N

                        B[Tg, 0] = B[Tg, 0] + dx * delta_phi
                        B[Tg, 1] = B[Tg, 1] + dy * delta_phi

                    if (bc_type == 'LIBRE'):
                        B[Tg, 0] = B[Tg, 0]
                        B[Tg, 1] = B[Tg, 1]

                        # Pour les arrêtes internes

                if neighbours_elements[1] != -1:
                    # Coordonnées X triangles

                    xtg = coordonnees_elements_x[neighbours_elements[0]]
                    xtd = coordonnees_elements_x[neighbours_elements[1]]

                    # Coordonnées Y triangles

                    ytg = coordonnees_elements_y[neighbours_elements[0]]
                    ytd = coordonnees_elements_y[neighbours_elements[1]]

                    # Récupération des dx et dy pour une arrête

                    dx = xtg - xtd
                    dy = ytg - ytd

                    # Pour les arêtes internes

                    ALS = np.zeros((2, 2))

                    # Création des différents paramèetres de la matrice 2x2

                    ALS[0, 0] = dx * dx
                    ALS[1, 0] = dx * dy
                    ALS[0, 1] = dy * dx
                    ALS[1, 1] = dy * dy

                    ATA[Tg] += ALS
                    ATA[Td] += ALS

                    # Remplissage de B

                    Phi_td = phi(xtd, ytd)  # Modification
                    Phi_tg = phi(xtg, ytg)  # Modification

                    B[Tg, 0] = B[Tg, 0] + (xtd - xtg) * (Phi_td - Phi_tg)
                    B[Tg, 1] = B[Tg, 1] + (ytd - ytg) * (Phi_td - Phi_tg)
                    B[Td, 0] = B[Td, 0] + (xtd - xtg) * (Phi_td - Phi_tg)
                    B[Td, 1] = B[Td, 1] + (ytd - ytg) * (Phi_td - Phi_tg)

            # Création de la matrice ATAI

            ATAI = np.zeros((f, 2, 2))

            # BOucle sur les triangles

            for i in range(number_of_elements):
                AL = ATA[i]  # Sélection de chaque élément
                ALI = np.linalg.inv(AL)  # Inversion de chaque élément
                ATAI[i] = ALI  # Ajout à la nouvelle matrice

            # Création du gradient

            Grad = np.zeros((f, 2))

            # Résolution numérique

            for i in range(number_of_elements):
                Grad[i] = np.dot(ATAI[i], B[i])  # Multiplication des deux matrices

            return Grad

        def reconstruction_least_square(mesh_obj, bcdata, phi, solution):

            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]

            # Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements

            # Création des coordonnées des éléments

            coordonnees_elements_x = []
            coordonnees_elements_y = []

            voisins = []

            f = mesh_obj.number_of_elements
            a = mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            bcdata = mesh_obj.get_boundary_faces_to_tag()

            # Création de la matrice de calcul ATA

            ATA = np.zeros((f, 2, 2))

            # Création de la matrice B

            B = np.zeros((f, 2))

            for i in range(number_of_elements):
                # Récupération des différents noeuds autour d'un éléments

                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]

                # Création du centre gémoétrique des différentes formes

                x_e, y_e = centre_element_2D(mesh_obj, i)

                # Récupération dans deux listes distinctes

                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)

                # Parcours de l'ensemble des arrêtes pour la création des différentes matrices

            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                # Récupération des éléments voisins
                voisins.append(neighbours_elements)

                Tg, Td = neighbours_elements

                if neighbours_elements[1] == -1:  # pour les faces frontières

                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers[tag]

                    if (bc_type == 'DIRICHLET'):
                        # DIRICHLET
                        # Création du centre de l'arrête
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)

                        # Milieu

                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])

                        x_milieu = (xa + xb) / 2
                        y_milieu = (yb + ya) / 2

                        # Dx et Dy
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        dx = (xtg - x_milieu)
                        dy = (ytg - y_milieu)

                        # Pour les arêtes internes

                        ALS = np.zeros((2, 2))

                        # Création des différents paramèetres de la matrice 2x2

                        ALS[0, 0] = dx * dx
                        ALS[1, 0] = dx * dy
                        ALS[0, 1] = dy * dx
                        ALS[1, 1] = dy * dy
                        # Remplissage

                        ATA[Tg] += ALS

                        Phi_A = phi(x_milieu, y_milieu)
                        Phi_tg = solution[Tg]  # Modifié pour la valeur précédente

                        B[Tg, 0] = B[Tg, 0] + (x_milieu - xtg) * (Phi_A - Phi_tg.item())
                        B[Tg, 1] = B[Tg, 1] + (y_milieu - ytg) * (Phi_A - Phi_tg.item())

                    if (bc_type == 'NEUMANN'):
                        # Neumann
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)

                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])

                        nx, ny = normal(xa, ya, xb, yb)

                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        x_milieu = (mesh_obj.get_node_to_xcoord(noeuds_faces[1]) + mesh_obj.get_node_to_xcoord(
                            noeuds_faces[0])) / 2
                        y_milieu = (mesh_obj.get_node_to_ycoord(noeuds_faces[1]) + mesh_obj.get_node_to_ycoord(
                            noeuds_faces[0])) / 2

                        dx1 = ((x_milieu - xtg))
                        dy1 = ((y_milieu - ytg))

                        dx = (dx1 * nx + dy1 * ny) * nx
                        dy = (dx1 * nx + dy1 * ny) * ny

                        # Matric intermédiaire
                        ALS = np.zeros((2, 2))

                        # Création des différents paramètres de la matrice 2x2

                        ALS[0, 0] = dx * dx
                        ALS[1, 0] = ALS[0, 1] = dy * dx
                        ALS[1, 1] = dy * dy

                        # Remplissage
                        ATA[Tg] += ALS

                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]

                        dx = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * nx
                        dy = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * ny

                        Phi_A = bc_number  # phi(x_milieu,y_milieu) # Phi milieu
                        Phi_N = bc_number

                        delta_phi = ((x_milieu - xtg) * nx + (y_milieu - ytg) * ny) * Phi_N

                        B[Tg, 0] = B[Tg, 0] + dx * delta_phi
                        B[Tg, 1] = B[Tg, 1] + dy * delta_phi

                    if (bc_type == 'LIBRE'):
                        B[Tg, 0] = B[Tg, 0]
                        B[Tg, 1] = B[Tg, 1]

                        # Pour les arrêtes internes

                if neighbours_elements[1] != -1:
                    # Coordonnées X triangles

                    xtg = coordonnees_elements_x[neighbours_elements[0]]
                    xtd = coordonnees_elements_x[neighbours_elements[1]]

                    # Coordonnées Y triangles

                    ytg = coordonnees_elements_y[neighbours_elements[0]]
                    ytd = coordonnees_elements_y[neighbours_elements[1]]

                    # Récupération des dx et dy pour une arrête

                    dx = xtg - xtd
                    dy = ytg - ytd

                    # Pour les arêtes internes

                    ALS = np.zeros((2, 2))

                    # Création des différents paramèetres de la matrice 2x2

                    ALS[0, 0] = dx * dx
                    ALS[1, 0] = dx * dy
                    ALS[0, 1] = dy * dx
                    ALS[1, 1] = dy * dy

                    ATA[Tg] += ALS
                    ATA[Td] += ALS

                    # Remplissage de B

                    Phi_td = solution[Td]  # Modification
                    Phi_tg = solution[Tg]  # Modification

                    B[Tg, 0] = B[Tg, 0] + (xtd - xtg) * (Phi_td.item() - Phi_tg.item())
                    B[Tg, 1] = B[Tg, 1] + (ytd - ytg) * (Phi_td.item() - Phi_tg.item())
                    B[Td, 0] = B[Td, 0] + (xtd - xtg) * (Phi_td.item() - Phi_tg.item())
                    B[Td, 1] = B[Td, 1] + (ytd - ytg) * (Phi_td.item() - Phi_tg.item())

            # Création de la matrice ATAI

            ATAI = np.zeros((f, 2, 2))

            # BOucle sur les triangles

            for i in range(number_of_elements):
                AL = ATA[i]  # Sélection de chaque élément
                ALI = np.linalg.inv(AL)  # Inversion de chaque élément
                ATAI[i] = ALI  # Ajout à la nouvelle matrice

            # Création du gradient

            Grad = np.zeros((f, 2))

            # Résolution numérique

            for i in range(number_of_elements):
                Grad[i] = np.dot(ATAI[i], B[i])  # Multiplication des deux matrices

            return Grad

        def analytique2(x, n):
            """Fonction
            Entrées:
                - prm : Objet class parametres()
                       - L : Longueur
                       - T_A : Temperature à x=A
                       - T_B : Temperature à x=B
                       - k : Conductivité thermique  W/m.K
                       - A : Section  m^2
                - N = Nombre de noeuds
            Sortie:
                - Vecteur (array) composé de la température en fonction de la position
                    en régime stationnaire de la solution analytique
            """
            # Paramètres
            L = 0.02  # Longueur (m)
            x_0 = 0  # Position initiale
            x_f = L  # Position finale
            T_A = 100  # Température à x = A (°C)
            T_B = 200  # Température à x = B (°C)
            k = 0.5  # Conductivité thermique (W/m·K)
            q = 1e6  # Source de chaleur (W/m³)

            # Résolution analytique
            T = np.zeros(n)
            for i in range(n):
                T[i] = ((T_B - T_A) / L + q / (2 * k) * (L - x[i])) * x[i] + T_A

            return T

        def erreur_L1(sol_num, sol_anal, dx):
            return np.sum(np.abs(sol_num - sol_anal) * dx) / len(sol_num)

        def erreur_L2(sol_num, sol_anal, dx):
            return np.sqrt(np.sum(dx * (sol_num - sol_anal) ** 2) / len(sol_num))

        def erreur_Linf(sol_num, sol_anal):
            return np.max(np.abs(sol_num - sol_anal))

        def OrdreConvergeance(x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            slope = dy / dx
            return slope

        # %% Graphique

        gamma = k
        S = q
        matrice = 0
        bcdata = (['DIRICHLET', T_B], ['NEUMANN', 0],
                  ['DIRICHLET', T_A], ['NEUMANN', 0])

        Grad = least_square(mesh_obj1, bcdata, phi)
        solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = solution_1
        for i in range(10):
            Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
            solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = np.reshape(solution, [Nx, Ny])
        Sol_coupe = solution[:, 0]

        # Affichage de champ scalaire avec pyvista
        nodes, elements = plotter.prepare_data_for_pyvista(mesh_obj1)
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Température'] = solution_1

        pl = pvQt.BackgroundPlotter()
        # Tracé du champ
        print("\nVoir le champ moyenne dans la fenêtre de PyVista \n")
        pl.add_mesh(pv_mesh, show_edges=True, scalars="Température", cmap="RdBu")

        # %% Comparaison avec solution analytique

        x = np.linspace(x_0 + dx / 2, x_f - dx / 2, Nx)
        T = Sol_coupe
        x_anal = np.linspace(x_0 + dx / 2, x_f - dx / 2, Nx)
        T_anal = analytique2(x_anal, Nx)
        plt.figure(1)
        plt.plot(x, T, '.', label='Solution numérique')
        plt.plot(x_anal, T_anal, label='Solution analytique')
        plt.title('Solutions du problème 4.2 à n=5')
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (°C)')
        plt.xlim(0, L)
        plt.legend()
        # %% Estimation de l'ordre de convergence
        n_values = np.arange(5, 21, 5)
        h_values = L / n_values
        Erreur_L1 = []
        Erreur_L2 = []
        Erreur_Linf = []
        for n in n_values:
            mesh_parameters1 = {'mesh_type': 'QUAD',
                                'Nx': n,
                                'Ny': n
                                }

            mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)
            conec = MeshConnectivity(mesh_obj1)
            conec.compute_connectivity()
            Grad = least_square(mesh_obj1, bcdata, phi)
            solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
            solution = solution_1
            for i in range(10):
                Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
                solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
            solution = np.reshape(solution, [n, n])
            Sol_coupe = solution[:, 0]
            dx2 = L / n
            x_e = np.linspace(L / n / 2, L - L / n / 2, n)
            sol_anal = analytique2(x_e, n)
            erreur_L1_value = erreur_L1(Sol_coupe, sol_anal, dx2)
            erreur_L2_value = erreur_L2(Sol_coupe, sol_anal, dx2)
            erreur_Linf_value = erreur_Linf(Sol_coupe, sol_anal)
            Erreur_L1.append(erreur_L1_value)
            Erreur_L2.append(erreur_L2_value)
            Erreur_Linf.append(erreur_Linf_value)

        plt.figure(4)
        plt.loglog(h_values, Erreur_L1, label="Erreur L1")
        plt.loglog(h_values, Erreur_L2, label="Erreur L2")
        plt.loglog(h_values, Erreur_Linf, label="Erreur Linfini")
        plt.xlabel("h")
        plt.ylabel("Erreur(h)")
        plt.legend()
        plt.grid(True)
        plt.title("Convergence de l'erreur du problème 4.2")
        plt.show()

        B = str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
        print("La convergeance de l'erreur pour le problème 4.2 est de :" + B)
        return
class MaillageTriangles():
    def __init__(self):
        import sympy as sp
        import numpy as np
        import pyvista as pv
        import pyvistaqt as pvQt
        from meshGenerator import MeshGenerator
        from meshConnectivity import MeshConnectivity
        from meshPlotter import MeshPlotter
        from mesh import Mesh
        import matplotlib.pyplot as plt

        #%% Paramètres
        L = 0.02        # Longueur (m)
        x_0 = 0         # Position initiale
        x_f = L         # Position finale
        T_A = 100       # Température à x = A (°C)
        T_B = 200       # Température à x = B (°C)
        k = 0.5         # Conductivité thermique (W/m·K)
        q = 1e6         # Source de chaleur (W/m³)

        Nx = 5          # Nombre de divisions en x
        Ny = 5          # Nombre de divisions en y
        lc = L/5   # Longueur caractéristique
        dx=L/Nx
          
        mesher = MeshGenerator()
        plotter = MeshPlotter()
        mesh_parameters1 = {'mesh_type': 'TRI','lc':lc}
                            

        mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)



        conec = MeshConnectivity(mesh_obj1)
        conec.compute_connectivity()

        #%% Fonctions utilitaires

        def normal(xa, ya, xb, yb):
            """
            Retourne les composantes a et b du vecteur normal unitaire à une arête.

            Parameters
            ----------
            xa, ya : float
                Coordonnées du point A.
            xb, yb : float
                Coordonnées du point B.

            Returns
            -------
            a, b : float
                Composantes du vecteur normal unitaire.
            """
            dx = xb - xa
            dy = yb - ya
            Delta_A = np.sqrt(dx**2 + dy**2)
            
            a = dy / Delta_A
            b = -dx / Delta_A
            
            return a, b

        def centre_element_2D(mesh_obj, i_element):
            """
            Calcule le centre géométrique d'un élément en 2D.

            Parameters
            ----------
            mesh_obj : object
                Objet maillage.
            i_element : int
                Index de l'élément.

            Returns
            -------
            xmoy, ymoy : float
                Coordonnées du centre géométrique de l'élément.
            """
            start = mesh_obj.get_element_to_nodes_start(i_element)
            fin = mesh_obj.get_element_to_nodes_start(i_element + 1)
            noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]
            
            xmoy = 0
            ymoy = 0
            for i in range(len(noeuds_i_elements)):
                x, y = mesh_obj.get_node_to_xycoord(noeuds_i_elements[i])
                xmoy += x
                ymoy += y

            xmoy /= len(noeuds_i_elements)
            ymoy /= len(noeuds_i_elements)
            
            return xmoy, ymoy

        def surface_cellule(noeuds, mesh_obj):
            """
            Calcule la surface d'une cellule à partir de ses nœuds.

            Parameters
            ----------
            noeuds : list
                Liste des nœuds de la cellule.
            mesh_obj : object
                Objet maillage.

            Returns
            -------
            surface : float
                Surface de la cellule.
            """
            n = len(noeuds)
            surface = 0
            points = []

            for i in range(n):
                x, y = mesh_obj.get_node_to_xycoord(noeuds[i])
                points.append((x, y))

            # Calcul de la surface (Formule du polygone - méthode du "shoelace")
            for i in range(n):
                x1, y1 = points[i]
                x2, y2 = points[(i + 1) % n]  # Boucle circulaire sur les points
                surface += x1 * y2 - x2 * y1

            return abs(surface) / 2

        def surface_moyenne(sol_num, mesh_obj):
            """
            Calcule la moyenne quadratique des surfaces des cellules.

            Parameters
            ----------
            sol_num : array
                Solution numérique.
            mesh_obj : object
                Objet maillage.

            Returns
            -------
            moyenne : float
                Moyenne quadratique des surfaces.
            """
            moyenne = 0
            
            for i in range(len(sol_num)):
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i + 1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin]
                
                taille_cellule = surface_cellule(noeuds_i_elements, mesh_obj)
                moyenne += taille_cellule**2
            
            moyenne = np.sqrt(moyenne / len(sol_num))
            
            return moyenne


        def diffusion2D(mesh_obj,bcdata,gradient,gamma,S,phi,matrice):
            
            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]
            
            # Création des coordonnées des éléments 
            
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            
            #Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements
            
            A = np.zeros((number_of_elements,number_of_elements))
            B = np.zeros((number_of_elements,1))
            t=np.zeros(number_of_elements)
            Taille=surface_moyenne(t, mesh_obj)
            B += S *Taille
            for i in range(number_of_elements):
                
                # Récupération des différents noeuds autour d'un éléments
                
                # start = mesh_obj.get_element_to_nodes_start(i)
                # fin = mesh_obj.get_element_to_nodes_start(i+1)
                # noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] 
                
                
                # Création du centre gémoétrique des différentes formes 
                
                x_e, y_e = centre_element_2D(mesh_obj, i)
                
                # Récupération dans deux listes distinctes 
                
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                
            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                nodes = mesh_obj.get_face_to_nodes(i)
                Tg,Td = neighbours_elements
                
                
                if neighbours_elements[1] != -1 :
                    
                    neighbours_elements = mesh_obj.get_face_to_elements(i)
                    nodes = mesh_obj.get_face_to_nodes(i)
                    Tg,Td = neighbours_elements
                    grad_tg = Grad[Tg]
                    grad_td = Grad[Td]
                    
                    # CRéation des points de la l'arrête
                    
                    xa,ya = mesh_obj.get_node_to_xycoord(nodes[0])
                    xb,yb = mesh_obj.get_node_to_xycoord(nodes[1])
                    
                    delta_x = xb - xa
                    delta_y = yb - ya
                    
                    # Création des points d'arrêtes 
                    element_A = neighbours_elements[0]
                    element_p = neighbours_elements[1]
                    
                    xA = coordonnees_elements_x[element_A]
                    xp = coordonnees_elements_x[element_p]
                    yA = coordonnees_elements_y[element_A]
                    yp = coordonnees_elements_y[element_p]
                    
                    # Création des paramètres 
                    delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2)
                    
                    delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
                    
                    delta_eta = delta_Ai
                    
                    nix,niy = normal(xa,ya,xb,yb)
                    
                    e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                    
                    e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                    
                    grad_b = gradient[Tg]
                    grad_a = gradient[Td]
                    # Construction des calculs 
                    
                    PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                    
                    PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                    
                    Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                    
                    if matrice == 0 :
                        # Uniquement si cela est connu 
                    
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)* (((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                    
                    if matrice == 1 :
                        
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)* (((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                    

                    t=np.zeros(number_of_elements)
                    Taille=surface_moyenne(t, mesh_obj)
                    # Implémentation des différents paramètres sur A et B 
                    
                    A[Tg][Tg] = A[Tg][Tg] + Di
                    
                    A[Td][Td] = A[Td][Td] + Di
                                
                    A[Tg][Td] = - Di + A[Tg][Td]
                    
                    A[Td][Tg] =  -Di + A[Td][Tg] 
                    
                    B[Tg] = B[Tg] + Sd_cross_i
                    
                    B[Td] = B[Td] - Sd_cross_i
            
                    
                if neighbours_elements[1] == -1 :
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers [tag]
                    
                    if bc_type == 'NEUMANN':
                        
                        neighbours_elements = mesh_obj.get_face_to_elements(i)
                        nodes = mesh_obj.get_face_to_nodes(i)
                        Tg,Td = neighbours_elements
                        
                        # CRéation des points de la l'arrête
                        
                        xa,ya = mesh_obj.get_node_to_xycoord(nodes[0])
                        xb,yb = mesh_obj.get_node_to_xycoord(nodes[1])
                        
                        delta_x = xb - xa
                        delta_y = yb - ya
                        
                        # Création des points d'arrêtes 
                        element_A = neighbours_elements[0]
                        element_p = neighbours_elements[1]
                        
                        xA = coordonnees_elements_x[element_A]
                        xp = coordonnees_elements_x[element_p]
                        yA = coordonnees_elements_y[element_A]
                        yp = coordonnees_elements_y[element_p]
                        
                        # Création des paramètres 
                        delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2)
                        
                        
                        delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
                        
                        delta_eta = delta_Ai
                        
                        nix,niy = normal(xa,ya,xb,yb)
                        
                        e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                        
                        e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                        
                        
                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]
                        
                        
                        # Construction des calculs 
                        
                        PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                        
                        PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                        
                        Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                        
                        if matrice == 0 :
                            # Uniquement si cela  n'est pas connu 
                        
                            Sd_cross_i = -gamma*(PKSIETA/PNKSI)*(((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                        
                        if matrice == 1 :
                            
                            Sd_cross_i = -gamma*(PKSIETA/PNKSI)*((phi(xb,yb)-phi(xa,ya))/delta_eta)* delta_Ai
                        
                        valeur = bc_number
                    
                        B[Tg] = B[Tg] + valeur*gamma*delta_Ai
                        
                        
                    if bc_type == 'DIRICHLET':
                        # Création des paramètres
                        
                        element_A = neighbours_elements[0]# Inutile
                        element_p = neighbours_elements[1]# Inutile
                        xa,ya = mesh_obj.get_node_to_xycoord(nodes[0]) # Inutile
                        xb,yb = mesh_obj.get_node_to_xycoord(nodes[1]) # Inutile
                        
                        xA = coordonnees_elements_x[element_A]
                        yA = coordonnees_elements_y[element_A]
                       
                        xp = (xa+xb)/2
                        yp = (ya+yb)/2
                        
                        delta_x = xb - xa
                        delta_y = yb - ya
                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]
                        
                        valeur = bc_number
                        
                        
                        delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2) #OK
                        
                        delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2) #OK
                        
                        delta_eta = delta_Ai #OK
                        
                        nix,niy = normal(xa,ya,xb,yb) #OK
                        
                        e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                        
                        e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                        
                        # Construction des calculs
                        
                        PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                        
                        PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                        
                        Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                        
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)*delta_Ai*(((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)
                        
                        A[Tg][Tg] = A[Tg][Tg] + Di
                        B[Tg] = B[Tg]+  Di*valeur +Sd_cross_i   #gamma*2*valeur 
                        
                    if (bc_type == 'LIBRE'):
                       B[Tg,0] = B[Tg,0] 
                       B[Tg,1] = B[Tg,1]
            solution = np.linalg.solve(A,B)
            
            return solution,A,B

        def diffusion2D_sansCross(mesh_obj,bcdata,gradient,gamma,S,phi,matrice):
            
            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]
            
            # Création des coordonnées des éléments 
            
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            
            #Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements
            
            A = np.zeros((number_of_elements,number_of_elements))
            B = np.zeros((number_of_elements,1))
            t=np.zeros(number_of_elements)
            Taille=surface_moyenne(t, mesh_obj)
            B += S*Taille
            for i in range(number_of_elements):
                
                # Récupération des différents noeuds autour d'un éléments
                
                # start = mesh_obj.get_element_to_nodes_start(i)
                # fin = mesh_obj.get_element_to_nodes_start(i+1)
                # noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] 
                
                
                # Création du centre gémoétrique des différentes formes 
                
                x_e, y_e = centre_element_2D(mesh_obj, i)
                
                # Récupération dans deux listes distinctes 
                
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                
            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                nodes = mesh_obj.get_face_to_nodes(i)
                Tg,Td = neighbours_elements
                
                
                if neighbours_elements[1] != -1 :
                    
                    neighbours_elements = mesh_obj.get_face_to_elements(i)
                    nodes = mesh_obj.get_face_to_nodes(i)
                    Tg,Td = neighbours_elements
                    grad_tg = Grad[Tg]
                    grad_td = Grad[Td]
                    
                    # CRéation des points de la l'arrête
                    
                    xa,ya = mesh_obj.get_node_to_xycoord(nodes[0])
                    xb,yb = mesh_obj.get_node_to_xycoord(nodes[1])
                    
                    delta_x = xb - xa
                    delta_y = yb - ya
                    
                    # Création des points d'arrêtes 
                    element_A = neighbours_elements[0]
                    element_p = neighbours_elements[1]
                    
                    xA = coordonnees_elements_x[element_A]
                    xp = coordonnees_elements_x[element_p]
                    yA = coordonnees_elements_y[element_A]
                    yp = coordonnees_elements_y[element_p]
                    
                    # Création des paramètres 
                    delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2)
                    
                    delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
                    
                    delta_eta = delta_Ai
                    
                    nix,niy = normal(xa,ya,xb,yb)
                    
                    e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                    
                    e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                    
                    grad_b = gradient[Tg]
                    grad_a = gradient[Td]
                    # Construction des calculs 
                    
                    PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                    
                    PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                    
                    Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                    
                    if matrice == 0 :
                        # Uniquement si cela est connu 
                    
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)* (((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                    
                    if matrice == 1 :
                        
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)* (((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                    

                    t=np.zeros(number_of_elements)
                    Taille=surface_moyenne(t, mesh_obj)
                    # Implémentation des différents paramètres sur A et B 
                    
                    A[Tg][Tg] = A[Tg][Tg] + Di
                    
                    A[Td][Td] = A[Td][Td] + Di
                                
                    A[Tg][Td] = - Di + A[Tg][Td]
                    
                    A[Td][Tg] =  -Di + A[Td][Tg] 
                    
                    B[Tg] = B[Tg]
                    
                    B[Td] = B[Td]
            
                    
                if neighbours_elements[1] == -1 :
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers [tag]
                    
                    if bc_type == 'NEUMANN':
                        
                        neighbours_elements = mesh_obj.get_face_to_elements(i)
                        nodes = mesh_obj.get_face_to_nodes(i)
                        Tg,Td = neighbours_elements
                        
                        # CRéation des points de la l'arrête
                        
                        xa,ya = mesh_obj.get_node_to_xycoord(nodes[0])
                        xb,yb = mesh_obj.get_node_to_xycoord(nodes[1])
                        
                        delta_x = xb - xa
                        delta_y = yb - ya
                        
                        # Création des points d'arrêtes 
                        element_A = neighbours_elements[0]
                        element_p = neighbours_elements[1]
                        
                        xA = coordonnees_elements_x[element_A]
                        xp = coordonnees_elements_x[element_p]
                        yA = coordonnees_elements_y[element_A]
                        yp = coordonnees_elements_y[element_p]
                        
                        # Création des paramètres 
                        delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2)
                        
                        
                        delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2)
                        
                        delta_eta = delta_Ai
                        
                        nix,niy = normal(xa,ya,xb,yb)
                        
                        e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                        
                        e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                        
                        
                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]
                        
                        
                        # Construction des calculs 
                        
                        PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                        
                        PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                        
                        Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                        
                        if matrice == 0 :
                            # Uniquement si cela  n'est pas connu 
                        
                            Sd_cross_i = -gamma*(PKSIETA/PNKSI)*(((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)* delta_Ai
                        
                        if matrice == 1 :
                            
                            Sd_cross_i = -gamma*(PKSIETA/PNKSI)*((phi(xb,yb)-phi(xa,ya))/delta_eta)* delta_Ai
                        
                        valeur = bc_number
                    
                        B[Tg] = B[Tg] + valeur*gamma*delta_Ai
                        
                        
                    if bc_type == 'DIRICHLET':
                        # Création des paramètres
                        
                        element_A = neighbours_elements[0]# Inutile
                        element_p = neighbours_elements[1]# Inutile
                        xa,ya = mesh_obj.get_node_to_xycoord(nodes[0]) # Inutile
                        xb,yb = mesh_obj.get_node_to_xycoord(nodes[1]) # Inutile
                        
                        xA = coordonnees_elements_x[element_A]
                        yA = coordonnees_elements_y[element_A]
                       
                        xp = (xa+xb)/2
                        yp = (ya+yb)/2
                        
                        delta_x = xb - xa
                        delta_y = yb - ya
                        grad_b = gradient[Tg]
                        grad_a = gradient[Td]
                        
                        valeur = bc_number
                        
                        
                        delta_ksi = ((xA-xp)**2 + (yA-yp)**2)**(1/2) #OK
                        
                        delta_Ai = ((xb-xa)**2+(yb-ya)**2)**(1/2) #OK
                        
                        delta_eta = delta_Ai #OK
                        
                        nix,niy = normal(xa,ya,xb,yb) #OK
                        
                        e_ksi = [ (xA-xp) /delta_ksi , (yA-yp)/delta_ksi]
                        
                        e_eta = [ (xb-xa) /delta_Ai  , (ya-yb)/delta_Ai]
                        
                        # Construction des calculs
                        
                        PNKSI = (delta_y * (xA-xp) / (delta_Ai * delta_ksi)) - (delta_x*(yA-yp) / (delta_Ai * delta_ksi))
                        
                        PKSIETA = ((xA-xp)*(xa-xb)/(delta_ksi * delta_eta)) + ((yA-yp)*(ya-yb)/(delta_ksi*delta_eta))
                        
                        Di = (1/PNKSI)*-gamma*delta_Ai/delta_ksi
                        
                        Sd_cross_i = -gamma*(PKSIETA/PNKSI)*delta_Ai*(((grad_a[0]+grad_b[0])/2)*(xb-xa) /delta_Ai)*(((grad_a[1]+grad_b[1])/2)*(yb-ya) /delta_Ai)
                        
                        A[Tg][Tg] = A[Tg][Tg] + Di
                        B[Tg] = B[Tg]+ Di*valeur
                        
                    if (bc_type == 'LIBRE'):
                       B[Tg,0] = B[Tg,0] 
                       B[Tg,1] = B[Tg,1]
            solution = np.linalg.solve(A,B)
            
            return solution,A,B

        def phi(x,y):
            return x 

        def least_square(mesh_obj,bcdata,phi):
            
            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]
            
            #Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements
            
            # Création des coordonnées des éléments 
            
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            
            voisins = []
            
            f = mesh_obj.number_of_elements
            a =mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            bcdata =mesh_obj.get_boundary_faces_to_tag()
            
            # Création de la matrice de calcul ATA 
            
            ATA = np.zeros((f,2,2))
            
            # Création de la matrice B 
            
            B = np.zeros((f,2))
            
            for i in range(number_of_elements):
                
                # Récupération des différents noeuds autour d'un éléments
                
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i+1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] 
                
                
                # Création du centre gémoétrique des différentes formes 
                
                x_e, y_e = centre_element_2D(mesh_obj, i)
                
                # Récupération dans deux listes distinctes 
                
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                
            # Parcours de l'ensemble des arrêtes pour la création des différentes matrices 
            
            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                # Récupération des éléments voisins
                voisins.append(neighbours_elements)
                
                Tg,Td = neighbours_elements
                
                if neighbours_elements[1] == -1 : # pour les faces frontières 
                    
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers [tag]
                    
                    
                    
                    if ( bc_type == 'DIRICHLET' ) :
                        # DIRICHLET
                        # Création du centre de l'arrête 
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        # Milieu
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        
                        x_milieu = (xa+xb)/2
                        y_milieu = (yb+ya)/2
                        
                        # Dx et Dy
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = (xtg - x_milieu)
                        dy = (ytg - y_milieu)
                        
                        # Pour les arêtes internes 
                        
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramèetres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= dx*dy
                        ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        # Remplissage
                        
                        ATA[Tg] += ALS
                        
                        Phi_A =  phi(x_milieu,y_milieu)
                        Phi_tg = phi(xtg,ytg) # Modifié pour la valeur précédente 
                        
                        B[Tg,0] = B[Tg,0] + (x_milieu - xtg) * (Phi_A - Phi_tg)
                        B[Tg,1] = B[Tg,1] + (y_milieu - ytg) * (Phi_A - Phi_tg)
                        
                        
                    if ( bc_type == 'NEUMANN' ) :
                        # Neumann
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        nx,ny = normal(xa,ya,xb,yb)
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]] 
                        
                        x_milieu = ( mesh_obj.get_node_to_xcoord(noeuds_faces[1]) + mesh_obj.get_node_to_xcoord(noeuds_faces[0]))/2
                        y_milieu = ( mesh_obj.get_node_to_ycoord(noeuds_faces[1]) + mesh_obj.get_node_to_ycoord(noeuds_faces[0]))/2
                        
                        dx1 =((x_milieu-xtg))
                        dy1 = ((y_milieu-ytg))
                        
                        dx = (dx1*nx+dy1*ny)*nx
                        dy = (dx1*nx + dy1*ny)*ny
                        
                        # Matric intermédiaire
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramètres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        
                        
                        
                        # Remplissage 
                        ATA[Tg] += ALS
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*nx
                        dy = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*ny
                        
                        Phi_A = bc_number # phi(x_milieu,y_milieu) # Phi milieu 
                        Phi_N = bc_number
                        
                        delta_phi = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*Phi_N
                        
                        B[Tg,0] = B[Tg,0] + dx * delta_phi
                        B[Tg,1] = B[Tg,1] + dy * delta_phi
                        
                        
                        
                    if (bc_type == 'LIBRE'):
                       B[Tg,0] = B[Tg,0] 
                       B[Tg,1] = B[Tg,1] 

                # Pour les arrêtes internes
                
                if neighbours_elements[1] != -1 :
                    
                    # Coordonnées X triangles 
                    
                    xtg = coordonnees_elements_x[neighbours_elements[0]]
                    xtd = coordonnees_elements_x[neighbours_elements[1]]
                    
                    # Coordonnées Y triangles
                    
                    ytg = coordonnees_elements_y[neighbours_elements[0]]
                    ytd = coordonnees_elements_y[neighbours_elements[1]]
                    
                    # Récupération des dx et dy pour une arrête
                    
                    dx = xtg - xtd
                    dy = ytg - ytd
                    
                    # Pour les arêtes internes
                    
                    ALS = np.zeros((2,2))
                    
                    # Création des différents paramèetres de la matrice 2x2
                    
                    ALS[0,0]= dx*dx
                    ALS[1,0]= dx*dy
                    ALS[0,1]= dy*dx
                    ALS[1,1]= dy*dy
                    
                    ATA[Tg]+= ALS
                    ATA[Td]+= ALS
                    
                    # Remplissage de B 
                    
                    Phi_td = phi(xtd,ytd)#Modification
                    Phi_tg = phi(xtg,ytg)#Modification 
                    
                    B[Tg,0] = B[Tg,0] + (xtd-xtg) * (Phi_td -Phi_tg)
                    B[Tg,1] = B[Tg,1] + (ytd-ytg) * (Phi_td -Phi_tg)
                    B[Td,0] = B[Td,0] + (xtd-xtg) * (Phi_td -Phi_tg)
                    B[Td,1] = B[Td,1] + (ytd-ytg) * (Phi_td -Phi_tg)
                    
            # Création de la matrice ATAI    
            
            ATAI = np.zeros((f,2,2))
            
            # BOucle sur les triangles 
            
            for i in range(number_of_elements):
                
                AL = ATA[i] # Sélection de chaque élément
                ALI = np.linalg.inv(AL) # Inversion de chaque élément
                ATAI[i]= ALI # Ajout à la nouvelle matrice
            
            # Création du gradient 
            
            Grad =np.zeros((f,2))
            
            # Résolution numérique 
            
            for i in range(number_of_elements):
                
                Grad[i] = np.dot(ATAI[i],B[i]) # Multiplication des deux matrices
                
            return Grad

        def reconstruction_least_square(mesh_obj,bcdata,phi,solution):
            
            bc_types = [item[0] for item in bcdata]  # Extrait la première valeur de chaque tuple
            bc_numbers = [item[1] for item in bcdata]
            
            #Création des données
            number_of_faces = mesh_obj.number_of_faces
            number_of_elements = mesh_obj.number_of_elements
            
            # Création des coordonnées des éléments 
            
            coordonnees_elements_x = []
            coordonnees_elements_y = []
            
            voisins = []
            
            f = mesh_obj.number_of_elements
            a =mesh_obj.number_of_faces
            s = mesh_obj.number_of_nodes
            bcdata =mesh_obj.get_boundary_faces_to_tag()
            
            # Création de la matrice de calcul ATA 
            
            ATA = np.zeros((f,2,2))
            
            # Création de la matrice B 
            
            B = np.zeros((f,2))
            
            for i in range(number_of_elements):
                
                # Récupération des différents noeuds autour d'un éléments
                
                start = mesh_obj.get_element_to_nodes_start(i)
                fin = mesh_obj.get_element_to_nodes_start(i+1)
                noeuds_i_elements = mesh_obj.element_to_nodes[start:fin] 
                
                
                # Création du centre gémoétrique des différentes formes 
                
                x_e, y_e = centre_element_2D(mesh_obj, i)
                
                # Récupération dans deux listes distinctes 
                
                coordonnees_elements_x.append(x_e)
                coordonnees_elements_y.append(y_e)  
                
            # Parcours de l'ensemble des arrêtes pour la création des différentes matrices 
            
            for i in range(number_of_faces):
                neighbours_elements = mesh_obj.get_face_to_elements(i)
                # Récupération des éléments voisins
                voisins.append(neighbours_elements)
                
                Tg,Td = neighbours_elements
                
                if neighbours_elements[1] == -1 : # pour les faces frontières 
                    
                    tag = mesh_obj.get_boundary_face_to_tag(i)
                    bc_type = bc_types[tag]
                    bc_number = bc_numbers [tag]
                    
                    
                    
                    if ( bc_type == 'DIRICHLET' ) :
                        # DIRICHLET
                        # Création du centre de l'arrête 
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        # Milieu
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        
                        x_milieu = (xa+xb)/2
                        y_milieu = (yb+ya)/2
                        
                        # Dx et Dy
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = (xtg - x_milieu)
                        dy = (ytg - y_milieu)
                        
                        # Pour les arêtes internes 
                        
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramèetres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= dx*dy
                        ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        # Remplissage
                        
                        ATA[Tg] += ALS
                        
                        Phi_A =  phi(x_milieu,y_milieu)
                        Phi_tg = solution[Tg] # Modifié pour la valeur précédente 
                        
                        B[Tg,0] = B[Tg,0] + (x_milieu - xtg) * (Phi_A - Phi_tg.item())
                        B[Tg,1] = B[Tg,1] + (y_milieu - ytg) * (Phi_A - Phi_tg.item())
                        
                        
                    if ( bc_type == 'NEUMANN' ) :
                        # Neumann
                        noeuds_faces = mesh_obj.get_face_to_nodes(i)
                        
                        xa = mesh_obj.get_node_to_xcoord(noeuds_faces[0])
                        ya = mesh_obj.get_node_to_ycoord(noeuds_faces[0])
                        xb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        yb = mesh_obj.get_node_to_xcoord(noeuds_faces[1])
                        
                        nx,ny = normal(xa,ya,xb,yb)
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]] 
                        
                        x_milieu = ( mesh_obj.get_node_to_xcoord(noeuds_faces[1]) + mesh_obj.get_node_to_xcoord(noeuds_faces[0]))/2
                        y_milieu = ( mesh_obj.get_node_to_ycoord(noeuds_faces[1]) + mesh_obj.get_node_to_ycoord(noeuds_faces[0]))/2
                        
                        dx1 =((x_milieu-xtg))
                        dy1 = ((y_milieu-ytg))
                        
                        dx = (dx1*nx+dy1*ny)*nx
                        dy = (dx1*nx + dy1*ny)*ny
                        
                        # Matric intermédiaire
                        ALS = np.zeros((2,2))
                        
                        # Création des différents paramètres de la matrice 2x2
                        
                        ALS[0,0]= dx*dx
                        ALS[1,0]= ALS[0,1]= dy*dx
                        ALS[1,1]= dy*dy
                        
                        
                        
                        # Remplissage 
                        ATA[Tg] += ALS
                        
                        xtg = coordonnees_elements_x[neighbours_elements[0]]
                        ytg = coordonnees_elements_y[neighbours_elements[0]]
                        
                        dx = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*nx
                        dy = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*ny
                        
                        Phi_A = bc_number # phi(x_milieu,y_milieu) # Phi milieu 
                        Phi_N = bc_number
                        
                        delta_phi = ((x_milieu-xtg)*nx + (y_milieu-ytg)*ny)*Phi_N
                        
                        B[Tg,0] = B[Tg,0] + dx * delta_phi
                        B[Tg,1] = B[Tg,1] + dy * delta_phi
                        
                        
                        
                    if (bc_type == 'LIBRE'):
                       B[Tg,0] = B[Tg,0] 
                       B[Tg,1] = B[Tg,1] 

                # Pour les arrêtes internes
                
                if neighbours_elements[1] != -1 :
                    
                    # Coordonnées X triangles 
                    
                    xtg = coordonnees_elements_x[neighbours_elements[0]]
                    xtd = coordonnees_elements_x[neighbours_elements[1]]
                    
                    # Coordonnées Y triangles
                    
                    ytg = coordonnees_elements_y[neighbours_elements[0]]
                    ytd = coordonnees_elements_y[neighbours_elements[1]]
                    
                    # Récupération des dx et dy pour une arrête
                    
                    dx = xtg - xtd
                    dy = ytg - ytd
                    
                    # Pour les arêtes internes
                    
                    ALS = np.zeros((2,2))
                    
                    # Création des différents paramèetres de la matrice 2x2
                    
                    ALS[0,0]= dx*dx
                    ALS[1,0]= dx*dy
                    ALS[0,1]= dy*dx
                    ALS[1,1]= dy*dy
                    
                    ATA[Tg]+= ALS
                    ATA[Td]+= ALS
                    
                    # Remplissage de B 
                    
                    Phi_td = solution[Td]#Modification
                    Phi_tg = solution[Tg]#Modification 
                    
                    B[Tg,0] = B[Tg,0] + (xtd-xtg) * (Phi_td.item() -Phi_tg.item())
                    B[Tg,1] = B[Tg,1] + (ytd-ytg) * (Phi_td.item() -Phi_tg.item())
                    B[Td,0] = B[Td,0] + (xtd-xtg) * (Phi_td.item() -Phi_tg.item())
                    B[Td,1] = B[Td,1] + (ytd-ytg) * (Phi_td.item() -Phi_tg.item())
                    
            # Création de la matrice ATAI    
            
            ATAI = np.zeros((f,2,2))
            
            # BOucle sur les triangles 
            
            for i in range(number_of_elements):
                
                AL = ATA[i] # Sélection de chaque élément
                ALI = np.linalg.inv(AL) # Inversion de chaque élément
                ATAI[i]= ALI # Ajout à la nouvelle matrice
            
            # Création du gradient 
            
            Grad =np.zeros((f,2))
            
            # Résolution numérique 
            
            for i in range(number_of_elements):
                
                Grad[i] = np.dot(ATAI[i],B[i]) # Multiplication des deux matrices
                
            return Grad

        def analytique2(x,n):
            """Fonction 
            Entrées:
                - prm : Objet class parametres()
                       - L : Longueur
                       - T_A : Temperature à x=A
                       - T_B : Temperature à x=B
                       - k : Conductivité thermique  W/m.K
                       - A : Section  m^2
                - N = Nombre de noeuds
            Sortie:
                - Vecteur (array) composé de la température en fonction de la position
                    en régime stationnaire de la solution analytique
            """
            #Paramètres
            L = 0.02        # Longueur (m)
            x_0 = 0         # Position initiale
            x_f = L         # Position finale
            T_A = 100       # Température à x = A (°C)
            T_B = 200       # Température à x = B (°C)
            k = 0.5         # Conductivité thermique (W/m·K)
            q = 1e6         # Source de chaleur (W/m³)
            

            # Résolution analytique
            T = np.zeros(n)
            for i in range(n):
                T[i]=((T_B-T_A)/L+q/(2*k)*(L-x[i]))*x[i]+T_A
            
            return T

        def erreur_L1(sol_num, sol_anal,dx):
            return np.sum(np.abs(sol_num - sol_anal)*dx)/len(sol_num)

        def erreur_L2(sol_num, sol_anal,dx):
            return np.sqrt(np.sum(dx*(sol_num - sol_anal)**2) / len(sol_num))

        def erreur_Linf(sol_num, sol_anal):
            return np.max(np.abs(sol_num - sol_anal))

        def OrdreConvergeance(x, y):
            dx = np.diff(x)
            dy = np.diff(y)
            slope = dy / dx
            return slope
        #%% Graphique

        gamma =k
        S=q
        matrice = 0
        bcdata = (['DIRICHLET', T_A ], ['NEUMANN', 0],
                  ['DIRICHLET', T_B], ['NEUMANN', 0])


        Grad = least_square(mesh_obj1, bcdata, phi)
        solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution_2 = diffusion2D_sansCross(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = solution_1
        solution2 = solution_2
        for _ in range(10):
            Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
            solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
            Grad2 = reconstruction_least_square(mesh_obj1, bcdata, phi, solution2)
            solution2 = diffusion2D_sansCross(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]


        # Initialize lists and arrays
        coordonnees_elements_x = []
        coordonnees_elements_y = []
        number_of_elements = mesh_obj1.number_of_elements
        Err = np.zeros([number_of_elements, 3])  # To store y values, x positions, and indices
        Err_y_val = []  # Use a list to store filtered results


        # Collect the center coordinates and store them in Err
        for i in range(number_of_elements):
            x_e, y_e = centre_element_2D(mesh_obj1, i)
            coordonnees_elements_x.append(x_e)
            coordonnees_elements_y.append(y_e)
            Err[i][0] = y_e  # Store y value
            Err[i][1] = x_e  # Store x position
            Err[i][2] = i     # Store index

        # Filter values within the specified range
        for element in Err:
            if 0.0035 <= element[0] <= 0.0045:  # Access y value directly
                Err_y_val.append((element[1], element[0], element[2]))  # Append x position, y value, and index

        # # Convert filtered values to a numpy array
        Err_y_val = np.array(Err_y_val)

        # Sort Err_y_val by the first column (x positions)
        Err_y_val_sorted = Err_y_val[np.argsort(Err_y_val[:, 0])]

        # # Assuming 'solutions' is your array containing solution values
        # # Initialize a list to store the corresponding solution values
        solution_values = []
        solution_values2 = []

        # # # Extract the corresponding solution values based on filtered indices
        for filtered_element in Err_y_val_sorted:
            index = int(filtered_element[2])  # Get the index of the element
            solution_values.append(solution_1[index])  # Append the solution value for that index
            solution_values2.append(solution_2[index])  # Append the solution value for that index

        # # Convert solution values to a numpy array if needed
        solution_values = np.array(solution_values)
        solution_values2 = np.array(solution_values2)
        # Affichage de champ scalaire avec pyvista
        nodes, elements = plotter.prepare_data_for_pyvista(mesh_obj1)
        pv_mesh = pv.PolyData(nodes, elements)
        pv_mesh['Température'] = solution_1

        pl = pvQt.BackgroundPlotter()
        # Tracé du champ
        print("\nVoir le champ moyenne dans la fenêtre de PyVista \n")
        pl.add_mesh(pv_mesh, show_edges=True, scalars="Température", cmap="RdBu")

        #%% Comparaison avec solution analytique
        plt.figure(1)
        x = np.linspace(x_0 + dx / 2, x_f - dx / 2, 6)
        T = solution_values
        x_anal = np.linspace(x_0 + dx / 2, x_f - dx / 2, 6)
        T_anal = analytique2(x_anal,6)
        plt.figure(1)
        plt.plot(x, T, '.', label='Solution numérique')
        plt.plot(x_anal, T_anal, label='Solution analytique' )
        plt.title('Solutions du problème 4.2')
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (°C)')
        plt.xlim(0,L)
        plt.legend()
        #%% Estimation de l'ordre de convergence 

        n_values = np.array([6, 13, 44])
        h_values = L / n_values
        Erreur_L1 = []  
        Erreur_L2 = []
        Erreur_Linf = []


        n = 6
        lc = 5
        mesh_parameters1 = {'mesh_type': 'TRI', 'lc': lc}
        mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)
        conec = MeshConnectivity(mesh_obj1)
        conec.compute_connectivity()
        Grad = least_square(mesh_obj1, bcdata, phi)
        solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = solution_1


        for _ in range(10):
            Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
            solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]


        dx2 = L / n
        x_e = np.linspace(L/n/2, L - L/n/2, n)
        sol_anal = analytique2(x_e, n)
        erreur_L1_value = erreur_L1(solution_1, sol_anal, dx2)
        erreur_L2_value = erreur_L2(solution_1, sol_anal, dx2)
        erreur_Linf_value = erreur_Linf(solution_1, sol_anal)
        Erreur_L1.append(erreur_L1_value)
        Erreur_L2.append(erreur_L2_value)
        Erreur_Linf.append(erreur_Linf_value)


        n = 13
        lc = 10
        mesh_parameters1 = {'mesh_type': 'TRI', 'lc': lc}
        mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)
        conec = MeshConnectivity(mesh_obj1)
        conec.compute_connectivity()
        Grad = least_square(mesh_obj1, bcdata, phi)
        solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = solution_1

        for _ in range(10):
            Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
            solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]


        dx2 = L / n
        x_e = np.linspace(L/n/2, L - L/n/2, n)
        sol_anal = analytique2(x_e, n)
        erreur_L1_value = erreur_L1(solution_1, sol_anal, dx2)
        erreur_L2_value = erreur_L2(solution_1, sol_anal, dx2)
        erreur_Linf_value = erreur_Linf(solution_1, sol_anal)
        Erreur_L1.append(erreur_L1_value)
        Erreur_L2.append(erreur_L2_value)
        Erreur_Linf.append(erreur_Linf_value)


        n = 44
        lc = 20
        mesh_parameters1 = {'mesh_type': 'TRI', 'lc': lc}
        mesh_obj1 = mesher.rectangle([0.0, L, 0.0, L], mesh_parameters1)
        conec = MeshConnectivity(mesh_obj1)
        conec.compute_connectivity()
        Grad = least_square(mesh_obj1, bcdata, phi)
        solution_1 = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]
        solution = solution_1

        for _ in range(10):
            Grad = reconstruction_least_square(mesh_obj1, bcdata, phi, solution)
            solution = diffusion2D(mesh_obj1, bcdata, Grad, gamma, S, phi, matrice)[0]



        dx2 = L / n
        x_e = np.linspace(L/n/2, L - L/n/2, n)
        sol_anal = analytique2(x_e, n)
        erreur_L1_value = erreur_L1(solution_1, sol_anal, dx2)
        erreur_L2_value = erreur_L2(solution_1, sol_anal, dx2)
        erreur_Linf_value = erreur_Linf(solution_1, sol_anal)
        Erreur_L1.append(erreur_L1_value)
        Erreur_L2.append(erreur_L2_value)
        Erreur_Linf.append(erreur_Linf_value)


        plt.figure(2)
        plt.loglog(h_values, Erreur_L1, label="Erreur L1")
        plt.loglog(h_values, Erreur_L2, label="Erreur L2")
        plt.loglog(h_values, Erreur_Linf, label="Erreur Linfini")
        plt.xlabel("h")
        plt.ylabel("Erreur(h)")
        plt.legend()
        plt.grid(True)
        plt.title("Convergence de l'erreur du problème 4.2")
        plt.show()


        B = str(max(abs(OrdreConvergeance(np.log(h_values), np.log(Erreur_Linf)))))
        print("La convergence de l'erreur pour le problème 4.2 est de :" + B)
        print("Cette erreur est due au gros ecart de valeur entre la solution analystique et celle trouvee avec un maillage triangulaire")

        #%% Comparaison sans cross diffusion 
        plt.figure(3)

        x2 = np.linspace(x_0 + dx / 2, x_f - dx / 2, 6)
        T2 = solution_values2
        plt.plot(x, T, '.', label='Solution numérique')
        plt.plot(x2, T2, label='Solution numérique sans cross diffusion' )
        plt.title('Solutions du problème 4.2')
        plt.xlabel('Position (m)')
        plt.ylabel('Temperature (°C)')
        plt.xlim(0,L)
        plt.legend()
        return
    


