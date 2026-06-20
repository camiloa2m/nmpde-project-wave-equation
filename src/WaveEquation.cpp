#include "WaveEquation.hpp"
#include <cmath>

void WaveEquation::setup()
{
  pcout << "===============================================" << std::endl;
  pcout << "Initializing the mesh" << std::endl;

  {
    Triangulation<dim> mesh_serial;
    const double left = 0.0;
    const double right = 1.0;

    if (dim == 1) {
      GridGenerator::subdivided_hyper_cube(mesh_serial, n_subdivisions, left, right, true);
    } else {
      // For simplicial meshes the 'colorize' option is not implemented in
      // deal.II's GridGenerator. Pass 'false' to avoid the ExcNotImplemented
      // exception raised when colorize==true.
      GridGenerator::subdivided_hyper_cube_with_simplices(mesh_serial, n_subdivisions, left, right, false);
    }

    GridTools::partition_triangulation(mpi_size, mesh_serial);
    const auto construction_data = TriangulationDescription::Utilities::
      create_description_from_triangulation(mesh_serial, MPI_COMM_WORLD);
    mesh.create_triangulation(construction_data);
    pcout << "  Number of elements = " << mesh.n_global_active_cells() << std::endl;
  }

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the finite element space" << std::endl;

  fe = std::make_unique<FE_SimplexP<dim>>(r);
  quadrature = std::make_unique<QGaussSimplex<dim>>(r + 1);
  mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the DoF handler" << std::endl;

  dof_handler.reinit(mesh);
  dof_handler.distribute_dofs(*fe);
  pcout << "  Number of DoFs = " << dof_handler.n_dofs() << std::endl;

  pcout << "-----------------------------------------------" << std::endl;
  pcout << "Initializing the linear system and vectors" << std::endl;

  const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  const IndexSet locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Use of constraints to manage Dirichlet boundary conditions efficiently, 
  // deal.II v9.6: Streamlined constraints.
  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  // Dirichlet Condition: u = g (Defined here strongly)
  Functions::ZeroFunction<dim> g; // Homogeneous Dirichlet BCs.
  // WaveEquation::FunctionG g; // Function<dim> providing boundary values
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0, // Boundary ID
                                            g, // value g
                                            constraints);
  constraints.close();
  // M and K are assembled WITHOUT constraints (pure FE matrices) and are used
  // only for matrix-vector products when building the RHS. The system matrices
  // matrix_u = M + theta^2*dt^2*K and matrix_v = M are assembled WITH the
  // Dirichlet constraints (via distribute_local_to_global), so each constrained
  // DOF gets a single identity row. The RHS gets constraints.set_zero each step.

  // Full pattern (no constraints) for the pure FE matrices M and K: they are
  // assembled with raw .add() and so need entries at constrained rows too.
  TrilinosWrappers::SparsityPattern sparsity_full(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_full);
  sparsity_full.compress();

  // Constrained pattern for the system matrices matrix_u and matrix_v, which are
  // assembled via distribute_local_to_global (condenses constrained entries).
  TrilinosWrappers::SparsityPattern sparsity_constrained(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_constrained, constraints, false);
  sparsity_constrained.compress();

  mass_matrix.reinit(sparsity_full);
  stiffness_matrix.reinit(sparsity_full);
  matrix_u.reinit(sparsity_constrained);
  matrix_v.reinit(sparsity_constrained);
  
  // Initialize all kinematic vectors
  solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  
  old_solution_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  old_solution.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);
  
  old_velocity_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  old_velocity.reinit(locally_owned_dofs, locally_relevant_dofs, MPI_COMM_WORLD);

  // Reuse temporary vectors for assembly and solving across timesteps to avoid reallocations.
  rhs_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  tmp_owned.reinit(locally_owned_dofs, MPI_COMM_WORLD);
  force_terms.reinit(locally_owned_dofs, MPI_COMM_WORLD);

  FunctionU0 u_0;
  FunctionV0 v_0;

  VectorTools::project(*mapping, dof_handler, constraints, QGaussSimplex<dim>(r + 2), u_0, old_solution_owned);
  VectorTools::project(*mapping, dof_handler, constraints, QGaussSimplex<dim>(r + 2), v_0, old_velocity_owned);

  old_solution = old_solution_owned;
  old_velocity = old_velocity_owned;
}

void WaveEquation::assemble_matrices()
{
  pcout << "Assembling matrices..." << std::endl;
  mass_matrix      = 0;
  stiffness_matrix = 0;
  matrix_u         = 0;
  matrix_v         = 0;

  FEValues<dim> fe_values(*fe, *quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
  const unsigned int dofs_per_cell = fe->dofs_per_cell;
  const unsigned int n_q = quadrature->size();
  FullMatrix<double> cell_mass(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_stiffness(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_u(dofs_per_cell, dofs_per_cell);
  FullMatrix<double> cell_v(dofs_per_cell, dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);

  // Coefficient combining M and K into the U-system matrix: M + theta^2*dt^2*K.
  const double u_stiffness_coeff = theta * theta * delta_t * delta_t;

  for (const auto &cell : dof_handler.active_cell_iterators()) {
    if (!cell->is_locally_owned()) continue;
    fe_values.reinit(cell);
    cell_mass = 0;
    cell_stiffness = 0;
    for (unsigned int q = 0; q < n_q; ++q) {
      const double rho_val = rho(fe_values.quadrature_point(q));
      const double c_val = c(fe_values.quadrature_point(q));
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {
          cell_mass(i, j) += rho_val * fe_values.shape_value(i, q) * fe_values.shape_value(j, q) * fe_values.JxW(q);
          cell_stiffness(i, j) += c_val * c_val * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q) * fe_values.JxW(q);
        }
      }
    }
    // Neumann BCs if needed can be added here.
    // Current implementation assumes homogeneous Neumann (natural) BCs, so no additional terms are added.

    // Local system matrices: U-system is M + theta^2*dt^2*K, V-system is M.
    cell_u = cell_mass;
    cell_u.add(u_stiffness_coeff, cell_stiffness);
    cell_v = cell_mass;

    cell->get_dof_indices(dof_indices);

    // M and K are assembled WITHOUT constraints: they are the pure FE matrices,
    // used only for matrix-vector products when building the RHS.
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          mass_matrix.add(dof_indices[i], dof_indices[j], cell_mass(i, j));
          stiffness_matrix.add(dof_indices[i], dof_indices[j], cell_stiffness(i, j));
        }

    // matrix_u and matrix_v are assembled WITH the Dirichlet constraints so each
    // constrained DOF gets a single identity row.
    // Local contributions are inserted into the global matrices consistently with 
    // the affine constraints.
    constraints.distribute_local_to_global(cell_u, dof_indices, matrix_u);
    constraints.distribute_local_to_global(cell_v, dof_indices, matrix_v);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
  matrix_u.compress(VectorOperation::add);
  matrix_v.compress(VectorOperation::add);
}

void WaveEquation::solve_timestep()
{
  const QGaussSimplex<dim> quadrature_rhs(r + 2);
  LambdaFunction ff(f);

  // Assemble forcing terms: forcing_terms = theta*dt*F(t_{n+1}) + (1-theta)*dt*F(t_n)
  // (forcing_terms is added to BOTH the U and V RHS)
  ff.set_time(time);
  VectorTools::create_right_hand_side(*mapping, dof_handler, quadrature_rhs, ff, force_terms);
  force_terms *= theta * delta_t;

  ff.set_time(time - delta_t);
  VectorTools::create_right_hand_side(*mapping, dof_handler, quadrature_rhs, ff, tmp_owned);
  force_terms.add((1.0 - theta) * delta_t, tmp_owned);

  // -----------------------------------------------------------------------
  // Step 1: solve for U^{n+1}
  // RHS = M*U^n + dt*M*V^n - theta*(1-theta)*dt^2*K*U^n + theta*dt*forcing_terms
  // System: (M + theta^2*dt^2*K) * U^{n+1} = RHS
  // -----------------------------------------------------------------------
  mass_matrix.vmult(rhs_owned, old_solution_owned);

  mass_matrix.vmult(tmp_owned, old_velocity_owned);
  rhs_owned.add(delta_t, tmp_owned);

  stiffness_matrix.vmult(tmp_owned, old_solution_owned);
  rhs_owned.add(-theta * (1.0 - theta) * delta_t * delta_t, tmp_owned);

  rhs_owned.add(theta * delta_t, force_terms);

  // matrix_u (= M + theta^2*dt^2*K, with Dirichlet identity rows) was assembled
  // once in assemble_matrices(). Zero the RHS at constrained DOFs so the system
  // is consistent with those identity rows (homogeneous Dirichlet => value 0).
  constraints.set_zero(rhs_owned);

  {
    TrilinosWrappers::PreconditionSSOR prec;
    prec.initialize(matrix_u);
    SolverControl sc(1000, 1e-8 * rhs_owned.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(sc);
    cg.solve(matrix_u, solution_owned, rhs_owned, prec);
    constraints.distribute(solution_owned);
    pcout << "  u-equation: " << sc.last_step() << " CG iterations." << std::endl;
  }

  solution = solution_owned;

  // -----------------------------------------------------------------------
  // Step 2: solve for V^{n+1}, using already-computed U^{n+1}
  // RHS = -theta*dt*K*U^{n+1} + M*V^n - (1-theta)*dt*K*U^n + forcing_terms
  // System: M * V^{n+1} = RHS
  // -----------------------------------------------------------------------
  stiffness_matrix.vmult(rhs_owned, solution_owned);
  rhs_owned *= -theta * delta_t;

  mass_matrix.vmult(tmp_owned, old_velocity_owned);
  rhs_owned += tmp_owned;

  stiffness_matrix.vmult(tmp_owned, old_solution_owned);
  rhs_owned.add(-(1.0 - theta) * delta_t, tmp_owned);

  rhs_owned += force_terms;

  // matrix_v (= M, with Dirichlet identity rows) was assembled once. Zero the
  // constrained RHS entries to keep the system consistent.
  constraints.set_zero(rhs_owned);

  {
    TrilinosWrappers::PreconditionSSOR prec;
    prec.initialize(matrix_v);
    SolverControl sc(1000, 1e-8 * rhs_owned.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(sc);
    cg.solve(matrix_v, velocity_owned, rhs_owned, prec);
    constraints.distribute(velocity_owned);
    pcout << "  v-equation: " << sc.last_step() << " CG iterations." << std::endl;
  }

  velocity = velocity_owned;

  // Shift to previous timestep
  old_solution_owned = solution_owned;
  old_velocity_owned = velocity_owned;
  old_solution       = solution_owned;
  old_velocity       = velocity_owned;
}

void WaveEquation::compute_energy() const 
{
  // Energy = 0.5 * V^T * M * V + 0.5 * U^T * K * U
  TrilinosWrappers::MPI::Vector tmp(old_solution_owned.locally_owned_elements(), MPI_COMM_WORLD);
  
  mass_matrix.vmult(tmp, old_velocity_owned);
  double kinetic = 0.5 * (old_velocity_owned * tmp);
  
  stiffness_matrix.vmult(tmp, old_solution_owned);
  double potential = 0.5 * (old_solution_owned * tmp);
  
  pcout << "  Energy: " << kinetic + potential << std::endl;
}


double
WaveEquation::compute_error(const VectorTools::NormType &norm_type,
                            const Function<dim>         &exact_solution) const
{
  const QGaussSimplex<dim> quadrature_error(r + 2);

  Vector<double> error_per_cell(mesh.n_active_cells());
  VectorTools::integrate_difference(*mapping,
                                    dof_handler,
                                    solution,
                                    exact_solution,
                                    error_per_cell,
                                    quadrature_error,
                                    norm_type);

  return VectorTools::compute_global_error(mesh, error_per_cell, norm_type);
}

double
WaveEquation::compute_energy_norm_error(const Function<dim> &exact_u,
                                         const Function<dim> &exact_v) const
{
  const QGaussSimplex<dim> quadrature_error(r + 2);
  FEValues<dim> fe_values(*mapping, *fe, quadrature_error,
                           update_values | update_gradients |
                           update_quadrature_points | update_JxW_values);

  const unsigned int n_q = quadrature_error.size();
  std::vector<Tensor<1, dim>> grad_uh(n_q);
  std::vector<double>         vh(n_q);
  std::vector<double>         exact_v_values(n_q);

  double potential_energy_error = 0.0; // 0.5 * int c^2 |grad(u - u_h)|^2
  double kinetic_energy_error   = 0.0; // 0.5 * int rho |v - v_h|^2

  for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      fe_values.reinit(cell);
      fe_values.get_function_gradients(solution, grad_uh);
      fe_values.get_function_values(velocity, vh);
      exact_v.value_list(fe_values.get_quadrature_points(), exact_v_values);

      for (unsigned int q = 0; q < n_q; ++q)
        {
          const double c_val   = c(fe_values.quadrature_point(q));
          const double rho_val = rho(fe_values.quadrature_point(q));

          const Tensor<1, dim> grad_e_u =
            exact_u.gradient(fe_values.quadrature_point(q)) - grad_uh[q];
          const double e_v = exact_v_values[q] - vh[q];

          potential_energy_error +=
            0.5 * c_val * c_val * (grad_e_u * grad_e_u) * fe_values.JxW(q);
          kinetic_energy_error += 0.5 * rho_val * e_v * e_v * fe_values.JxW(q);
        }
    }

  const double local_energy_error_sq = potential_energy_error + kinetic_energy_error;
  const double global_energy_error_sq =
    Utilities::MPI::sum(local_energy_error_sq, MPI_COMM_WORLD);

  return std::sqrt(global_energy_error_sq);
}


void WaveEquation::output() const
{
  DataOut<dim> data_out;
  data_out.add_data_vector(dof_handler, solution, "displacement");
  data_out.add_data_vector(dof_handler, velocity, "velocity");

  std::vector<unsigned int> partition_int(mesh.n_active_cells());
  GridTools::get_subdomain_association(mesh, partition_int);
  const Vector<double> partitioning(partition_int.begin(), partition_int.end());
  data_out.add_data_vector(partitioning, "partitioning");

  data_out.build_patches();

  const std::filesystem::path mesh_path(mesh_file_name);

  // One directory per mesh size: <output_dir>/N4, <output_dir>/N8, ...
  const std::filesystem::path out_dir =
    std::filesystem::path(output_dir) / ("N" + std::to_string(n_subdivisions));

  std::filesystem::create_directories(out_dir);

  const std::string output_file_name = "output-" + mesh_path.stem().string();

  data_out.write_vtu_with_pvtu_record(out_dir.string() + "/", output_file_name, timestep_number, MPI_COMM_WORLD);
}

void WaveEquation::run()
{
  setup();
  // M and K are constant in time for this problem, so we assemble them once.
  // If the problem had time-dependent coefficients or nonlinear terms appear, 
  // we would need to reassemblethese matrices at each timestep.
  assemble_matrices();

  if (enable_output)
    output(); // Output initial state (t=0)
  compute_energy();

  while (time < T - 0.5 * delta_t) {
    time += delta_t;
    ++timestep_number;

    pcout << "Timestep " << timestep_number << " at t = " << time << std::endl;
    solve_timestep();
    compute_energy();

    // Output periodically to save disk space if delta_t is very small
    if (enable_output && timestep_number % 2 == 0) {
      output();
    }
  }
}