TEMPLATE_RES_REDOCK = '''Docking results from DINC-Ensemble\n \
\n \
Input ligand: {LIGAND}\n \
Input receptors: {RECEPTORS}\n \
Job type: redock\n \
\n \
**************************************\n \
+ Conformations sorted by increasing binding energy:\n \
----------------------------------------------------------------------------------------------------\n \
         File name               Energy (kcal/mol)       RMSD (A)       Receptor\n \
----------------------------------------------------------------------------------------------------\n\
{ENERGY_CONF_TXT}\n \
----------------------------------------------------------------------------------------------------\n \
\n \
+ Conformations sorted by decreasing cluster size\n \
(lowest-energy conformation from each cluster):\n \
----------------------------------------------------------------------------------------------------\n \
         File name               Energy (kcal/mol)       RMSD (A)       Receptor\n \
----------------------------------------------------------------------------------------------------\n\
{CLUST_CONF_TXT}\n \
----------------------------------------------------------------------------------------------------\n \
\n \
+ Conformations sorted by increasing RMSD:\n \
----------------------------------------------------------------------------------------------------\n \
         File name               Energy (kcal/mol)       RMSD (A)       Receptor\n \
----------------------------------------------------------------------------------------------------\n\
{RMSD_CONF_TXT}\n \
----------------------------------------------------------------------------------------------------\n \
Note: RMSD values are computed with respect to the\n \
      input conformation of the ligand\n \
\n \
Total runtime = {RUNTIME}\n \

Thank you for using DINC-Ensemble!\n \
'''

TEMPLATE_RES_CROSSDOCK = "Docking results from DINC-Ensemble\n \
\n \
Input ligand: {LIGAND}\n \
Input receptors: {RECEPTORS}\n \
Job type: crossdock\n \
\n \
**************************************\n \
+ Conformations sorted by increasing binding energy:\n \
----------------------------------------------------------------------------------------------------\n \
         File name               Energy (kcal/mol)       RMSD (A)       Receptor\n \
----------------------------------------------------------------------------------------------------\n\
{ENERGY_CONF_TXT}\n \
----------------------------------------------------------------------------------------------------\n \
\n \
+ Conformations sorted by decreasing cluster size\n \
(lowest-energy conformation from each cluster):\n \
----------------------------------------------------------------------------------------------------\n \
         File name               Energy (kcal/mol)       RMSD (A)       Receptor\n \
----------------------------------------------------------------------------------------------------\n\
{CLUST_CONF_TXT}\n \
--------------------------------------------------------------------------------------------------\n \
Note: RMSD values are computed with respect to the\n \
      lowest energy conformation of the ligand\n \
\n \
Total runtime = {RUNTIME}\n \
\n \
Thank you for using DINC-Ensemble!\n \
"


TEMPLATE_NO_RES = '''Docking results from DINC-Ensemble\n \
\n \
Input ligand: {LIGAND}\n \
Input receptors: {RECEPTORS}\n \
\n \
**************************************\n \
No results found - check the log file for errors!\n \
**************************************\n \
\n \
\n \
Total runtime = {RUNTIME}\n \
\n \
Thank you for using DINC-Ensemble!\n \
'''