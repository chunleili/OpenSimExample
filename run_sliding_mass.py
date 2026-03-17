import opensim as osim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

print('=== Creating sliding mass model ===')
model = osim.Model()
model.setName('sliding_mass')
model.set_gravity(osim.Vec3(0, 0, 0))
body = osim.Body('body', 2.0, osim.Vec3(0), osim.Inertia(0))
model.addComponent(body)

joint = osim.SliderJoint('slider', model.getGround(), body)
coord = joint.updCoordinate()
coord.setName('position')
model.addComponent(joint)

actu = osim.CoordinateActuator()
actu.setCoordinate(coord)
actu.setName('actuator')
actu.setOptimalForce(1)
model.addComponent(actu)

body.attachGeometry(osim.Sphere(0.05))
model.finalizeConnections()
model.printToXML('SlidingMass.osim')
print('Model saved to SlidingMass.osim')

print('=== Setting up MocoStudy ===')
study = osim.MocoStudy()
study.setName('sliding_mass')

problem = study.updProblem()
problem.setModel(model)
problem.setTimeBounds(osim.MocoInitialBounds(0.), osim.MocoFinalBounds(0., 5.))
problem.setStateInfo('/slider/position/value', osim.MocoBounds(-5, 5),
                     osim.MocoInitialBounds(0), osim.MocoFinalBounds(1))
problem.setStateInfo('/slider/position/speed', [-50, 50], [0], [0])
problem.setControlInfo('/actuator', osim.MocoBounds(-50, 50))
problem.addGoal(osim.MocoFinalTimeGoal())

solver = study.initCasADiSolver()
solver.set_num_mesh_intervals(100)
study.printToXML('sliding_mass.omoco')
print('Study saved to sliding_mass.omoco')

print('=== Solving... ===')
solution = study.solve()
solution.write('sliding_mass_solution.sto')
print('Solution saved to sliding_mass_solution.sto')

solutionAsTable = solution.exportToStatesTable()
times = solutionAsTable.getIndependentColumn()
positions = solutionAsTable.getDependentColumn('/slider/position/value')
speeds = solutionAsTable.getDependentColumn('/slider/position/speed')

plt.figure(figsize=(10, 6))
plt.title('Position and Speed of the mass over time')
plt.plot(times, positions.to_numpy(), label='Position')
plt.plot(times, speeds.to_numpy(), label='Speed')
plt.legend(loc='best')
plt.xlabel('Time')
plt.ylabel('Value')
plt.savefig('sliding_mass_result.png', dpi=150)
print('Plot saved to sliding_mass_result.png')
print('=== Done! ===')
