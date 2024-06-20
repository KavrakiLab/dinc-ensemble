from os import path
from subprocess import call
from threading import Thread

from AutoDockTools.MoleculePreparation import AD4LigandWriter


##
# The DockingThread class allows running multiple docking jobs in parallel using multi-threading.
#   fragment: name of the fragment to be docked, in the format <ligand_name>_frag_<X>_conf_<Y>
#   receptor: name of the file containing the protein receptor
#   flex: name of the file containing the flexible residues of the protein receptor
#   params: parameters of the docking job
#   randomized: should the initial conformation of the ligand be randomized during docking?
#   center: position of the ligand's center, used by AD4 when the docking is not randomized
##
class DockingThread(Thread):
    def __init__(self, fragment, receptor, flex, params, randomized=True, center=[]):
        self.fragment = fragment
        self.flex = flex
        self.receptor = receptor
        self.param = params
        self.randomized = randomized
        self.center = center
        Thread.__init__(self)

    # Submit the docking job
    def run(self):
        print(" - docking thread for: " + path.basename(self.fragment))
        dir_path = path.dirname(self.fragment)
        ligand = self.fragment + ".pdbqt"

        # docking with AutoDock4
        if self.param["sampling"] == "AD4":
            # call the python script that prepares the docking job
            if self.param["flex"]:
                command = (
                    "prepare_dpf42.py -l "
                    + ligand
                    + " -r "
                    + self.receptor
                    + " -x "
                    + self.flex
                    + " -o "
                    + self.fragment
                    + ".dpf"
                    + " -p ga_num_evals="
                    + str(self.param["ga_num_evaluations"])
                    + " -p ga_run="
                    + str(self.param["ga_run"])
                    + " -p ga_pop_size="
                    + str(self.param["ga_pop_size"])
                    + " -p ga_num_generations="
                    + str(self.param["ga_num_generations"])
                )
            else:
                command = (
                    "prepare_dpf42.py -l "
                    + ligand
                    + " -r "
                    + self.receptor
                    + " -o "
                    + self.fragment
                    + ".dpf"
                    + " -p ga_num_evals="
                    + str(self.param["ga_num_evaluations"])
                    + " -p ga_run="
                    + str(self.param["ga_run"])
                    + " -p ga_pop_size="
                    + str(self.param["ga_pop_size"])
                    + " -p ga_num_generations="
                    + str(self.param["ga_num_generations"])
                )
            # if not self.randomized:
            # add the ligand's center to the definition of the docking job
            # command += " -p tran0=" + str(self.center)[1:-1].replace(',', '')
            # command += " -p axisangle0=0.0 0.0 0.0 0.0"
            call(command, shell=True, cwd=dir_path)

            # call AutoDock4 to perform the docking
            call(
                "autodock4 -p {0}.dpf -l {0}.dlg".format(self.fragment),
                shell=True,
                cwd=dir_path,
            )

        # docking with AutoDock Vina
        elif self.param["sampling"] == "Vina":
            with open(self.fragment + ".log", "w") as log:
                config = path.join(dir_path, self.fragment + ".cfg")
                prepare_vina_config_file(config, ligand, self.receptor, self.param)
                if self.param["flex"]:
                    call(
                        "vina --config %s --flex %s --cpu %s --exhaustiveness %s --num_modes %s \
                        --energy_range %s"
                        % (
                            config,
                            self.flex,
                            self.param["num_cpu"],
                            self.param["exhaustiveness"],
                            self.param["num_modes"],
                            self.param["energy_range"],
                        ),
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )
                else:
                    call(
                        "vina --config %s --cpu %s --exhaustiveness %s --num_modes %s --energy_range %s"
                        % (
                            config,
                            self.param["num_cpu"],
                            self.param["exhaustiveness"],
                            self.param["num_modes"],
                            self.param["energy_range"],
                        ),
                        stdout=log,
                        shell=True,
                        cwd=dir_path,
                    )


# Create randomized conformations of a ligand, and dock them in parallel.
#   ligand_file: name of the file containing the ligand
#   conf_list: names of the ligand conformers that will be randomly created and docked
#   receptor_file: name of the file containing the protein receptor
#   params: parameters of the docking job
#   dir_path: directory where the docking job is running
#   flex_file: name of the file containing the flexible residues of the protein receptor
#
def randomize_and_dock(
    ligand_file, conf_list, receptor_file, params, dir_path, flex_file=None
):
    # create randomized conformations for the given ligand
    with open(path.join(dir_path, "randomize.log"), "w") as log:
        config_file = path.join(dir_path, "randomize_ligand.cfg")
        prepare_vina_config_file(config_file, ligand_file, receptor_file, params)
        for conf in conf_list:
            call(
                "vina --randomize_only --cpu 1 --config %s --out %s.pdbqt"
                % (config_file, conf),
                stdout=log,
                shell=True,
                cwd=dir_path,
            )

    # dock the various ligand conformers in multiple parallel threads
    threads = []
    for ligand_conf in conf_list:
        t = DockingThread(ligand_conf, receptor_file, flex_file, params)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()


# Write file into the initial fragment file before running randomize and dock
def write_randomize_dock(
    mol_frags,
    initial_frag_file,
    conf_frag_list,
    receptor_file,
    params,
    dir_path,
    flex_file,
):
    AD4LigandWriter().write(mol_frags[0], initial_frag_file)
    randomize_and_dock(
        initial_frag_file, conf_frag_list, receptor_file, params, dir_path, flex_file
    )


# Create a configuration file with the given name (config_file_name) for docking with Vina.
#   ligand_file: name of the file containing the ligand
#   receptor_file: name of the file containing the protein receptor
#   params: parameters of the docking job
#
def prepare_vina_config_file(config_file_name, ligand_file, receptor_file, params):
    (x, y, z) = params["box_center"]
    (X, Y, Z) = params["box"]
    with open(config_file_name, "w") as config_file:
        config_file.write("receptor = " + receptor_file + "\n")
        config_file.write("ligand = " + ligand_file + "\n")
        config_file.write("center_x = " + str(x) + "\n")
        config_file.write("center_y = " + str(y) + "\n")
        config_file.write("center_z = " + str(z) + "\n")
        config_file.write("size_x = " + str(X) + "\n")
        config_file.write("size_y = " + str(Y) + "\n")
        config_file.write("size_z = " + str(Z) + "\n")
