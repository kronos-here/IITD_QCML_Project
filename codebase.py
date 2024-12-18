import qiskit
import qiskit_nature
import pyscf

from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver

# Step 1: Define the molecule (Formic Acid - HCOOH)
driver = PySCFDriver(atom='H 0 0 0; C 0 0 1.1; O 0 0 2.3; O 0 1.2 1.1; H 0 1.2 0',
                     basis='sto3g')

es_problem = driver.run()

from qiskit_nature.second_q.mappers import JordanWignerMapper

mapper = JordanWignerMapper()

from qiskit_algorithms import NumPyMinimumEigensolver

numpy_solver = NumPyMinimumEigensolver()

from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.primitives import Estimator
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

#ansatz preparation takes lot of time.

ansatz = UCCSD(
    es_problem.num_spatial_orbitals,
    es_problem.num_particles,
    mapper,
    initial_state=HartreeFock(
        es_problem.num_spatial_orbitals,
        es_problem.num_particles,
        mapper,
    ),
)

vqe_solver = VQE(Estimator(), ansatz, SLSQP())

vqe_solver.initial_point = [0.0] * ansatz.num_parameters

from qiskit_algorithms import VQE
from qiskit.circuit.library import TwoLocal

tl_circuit = TwoLocal(
    rotation_blocks=["h", "rx"],
    entanglement_blocks="cz",
    entanglement="full",
    reps=2,
    parameter_prefix="y",
)

another_solver = VQE(Estimator(), tl_circuit, SLSQP())

from qiskit_nature.second_q.algorithms import GroundStateEigensolver

calc = GroundStateEigensolver(mapper, vqe_solver)

res = calc.solve(es_problem)
print(res)