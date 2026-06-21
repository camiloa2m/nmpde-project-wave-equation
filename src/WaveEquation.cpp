#include "WaveEquation.hpp"
#include <cmath>
#include <iomanip>

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

  constraints.clear();
  DoFTools::make_hanging_node_constraints(dof_handler, constraints);
  Functions::ZeroFunction<dim> g; // Homogeneous Dirichlet BCs.
  std::map<types::global_dof_index, double> boundary_values_map;
  VectorTools::interpolate_boundary_values(dof_handler,
                                            0, // Boundary ID
                                            g, // value g
                                            boundary_values_map);
  for (const auto &entry : boundary_values_map)
    constraints.add_line(entry.first);
  constraints.close();
  // M and K are assembled WITHOUT constraints (pure FE matrices), used only
  // for matrix-vector products when building the RHS. matrix_u and matrix_v
  // are assembled WITH the Dirichlet constraints (via
  // distribute_local_to_global), giving each constrained DOF a single
  // identity row. The constraint PATTERN is fixed here and reused every
  // timestep, but since boundary_g/boundary_dgdt can be time-dependent
  // while matrix_u/matrix_v are assembled only once, the prescribed VALUE
  // is written into the RHS directly each step instead (see solve_timestep()).

  // Record (dof, support point) for every locally-owned Dirichlet-boundary
  // DOF, so solve_timestep() can evaluate boundary_g/boundary_dgdt there
  // without repeating the boundary search.
  {
    std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler, support_points);
    boundary_dofs.clear();
    for (const auto &entry : boundary_values_map)
      if (locally_owned_dofs.is_element(entry.first))
        boundary_dofs.emplace_back(entry.first, support_points[entry.first]);
  }

  // Full pattern (no constraints) for M and K: assembled with raw .add(),
  // so they need entries at constrained rows too.
  TrilinosWrappers::SparsityPattern sparsity_full(locally_owned_dofs, MPI_COMM_WORLD);
  DoFTools::make_sparsity_pattern(dof_handler, sparsity_full);
  sparsity_full.compress();

  // Constrained pattern for matrix_u and matrix_v, assembled via
  // distribute_local_to_global (condenses constrained entries).
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

  // The projection above used the homogeneous-Dirichlet `constraints`
  // object, so boundary DOFs are left at 0. Overwrite them with the
  // prescribed g(x,0)/dg_dt(x,0) so the initial state matches the
  // boundary data solve_timestep() uses from t=0 onward.
  if (boundary_g)
    for (const auto &[dof, point] : boundary_dofs)
      old_solution_owned(dof) = boundary_g(point, 0.0);
  if (boundary_dgdt)
    for (const auto &[dof, point] : boundary_dofs)
      old_velocity_owned(dof) = boundary_dgdt(point, 0.0);
  old_solution_owned.compress(VectorOperation::insert);
  old_velocity_owned.compress(VectorOperation::insert);

  old_solution = old_solution_owned;
  old_velocity = old_velocity_owned;
}

void WaveEquation::assemble_matrices()
{
  TimerOutput::Scope t(computing_timer, "assembly");

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
    // Homogeneous Neumann (natural) BCs, so no boundary term to add here.

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

    // matrix_u and matrix_v are assembled WITH the Dirichlet constraints, so
    // each constrained DOF gets a single identity row.
    constraints.distribute_local_to_global(cell_u, dof_indices, matrix_u);
    constraints.distribute_local_to_global(cell_v, dof_indices, matrix_v);
  }

  mass_matrix.compress(VectorOperation::add);
  stiffness_matrix.compress(VectorOperation::add);
  matrix_u.compress(VectorOperation::add);
  matrix_v.compress(VectorOperation::add);
}

void WaveEquation::apply_dirichlet_value(
  const std::vector<std::pair<types::global_dof_index, Point<dim>>> &dofs,
  const std::function<double(const Point<dim> &, const double &)>   &value_fn,
  const double                                                        t,
  TrilinosWrappers::MPI::Vector                                       &rhs)
{
  for (const auto &[dof, point] : dofs)
    rhs(dof) = value_fn ? value_fn(point, t) : 0.0;
  rhs.compress(VectorOperation::insert);
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

  // Step 1: solve for U^{n+1}.
  // RHS = M*U^n + dt*M*V^n - theta*(1-theta)*dt^2*K*U^n + theta*dt*forcing_terms
  // System: (M + theta^2*dt^2*K) * U^{n+1} = RHS
  mass_matrix.vmult(rhs_owned, old_solution_owned);

  mass_matrix.vmult(tmp_owned, old_velocity_owned);
  rhs_owned.add(delta_t, tmp_owned);

  stiffness_matrix.vmult(tmp_owned, old_solution_owned);
  rhs_owned.add(-theta * (1.0 - theta) * delta_t * delta_t, tmp_owned);

  rhs_owned.add(theta * delta_t, force_terms);

  // matrix_u has Dirichlet identity rows (assembled once in assemble_matrices());
  // overwrite the RHS at those DOFs with U's boundary value at t^{n+1} = time.
  apply_dirichlet_value(boundary_dofs, boundary_g, time, rhs_owned);

  {
    TimerOutput::Scope t(computing_timer, "solve U");
    TrilinosWrappers::PreconditionSSOR prec;
    prec.initialize(matrix_u);
    SolverControl sc(1000, 1e-8 * rhs_owned.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(sc);
    cg.solve(matrix_u, solution_owned, rhs_owned, prec);
    // The identity rows already force solution_owned == rhs_owned at
    // boundary DOFs up to CG tolerance; re-impose the exact value so that
    // tolerance noise doesn't leak into the V-solve or error norms.
    apply_dirichlet_value(boundary_dofs, boundary_g, time, solution_owned);
    pcout << "  u-equation: " << sc.last_step() << " CG iterations." << std::endl;
  }

  solution = solution_owned;

  // Step 2: solve for V^{n+1}, using the already-computed U^{n+1}.
  // RHS = -theta*dt*K*U^{n+1} + M*V^n - (1-theta)*dt*K*U^n + forcing_terms
  // System: M * V^{n+1} = RHS
  stiffness_matrix.vmult(rhs_owned, solution_owned);
  rhs_owned *= -theta * delta_t;

  mass_matrix.vmult(tmp_owned, old_velocity_owned);
  rhs_owned += tmp_owned;

  stiffness_matrix.vmult(tmp_owned, old_solution_owned);
  rhs_owned.add(-(1.0 - theta) * delta_t, tmp_owned);

  rhs_owned += force_terms;

  // matrix_v (= M, with Dirichlet identity rows) was assembled once too.
  // V = dU/dt, so its boundary condition is dg/dt(t^{n+1}), not g(t^{n+1}).
  apply_dirichlet_value(boundary_dofs, boundary_dgdt, time, rhs_owned);

  {
    TimerOutput::Scope t(computing_timer, "solve V");
    TrilinosWrappers::PreconditionSSOR prec;
    prec.initialize(matrix_v);
    SolverControl sc(1000, 1e-8 * rhs_owned.l2_norm());
    SolverCG<TrilinosWrappers::MPI::Vector> cg(sc);
    cg.solve(matrix_v, velocity_owned, rhs_owned, prec);
    apply_dirichlet_value(boundary_dofs, boundary_dgdt, time, velocity_owned);
    pcout << "  v-equation: " << sc.last_step() << " CG iterations." << std::endl;
  }

  velocity = velocity_owned;

  // Shift to previous timestep
  old_solution_owned = solution_owned;
  old_velocity_owned = velocity_owned;
  old_solution       = solution_owned;
  old_velocity       = velocity_owned;
}

double WaveEquation::compute_energy()
{
  // Energy = 0.5 * V^T * M * V + 0.5 * U^T * K * U
  TrilinosWrappers::MPI::Vector tmp(old_solution_owned.locally_owned_elements(), MPI_COMM_WORLD);

  mass_matrix.vmult(tmp, old_velocity_owned);
  double kinetic = 0.5 * (old_velocity_owned * tmp);

  stiffness_matrix.vmult(tmp, old_solution_owned);
  double potential = 0.5 * (old_solution_owned * tmp);

  const double energy = kinetic + potential;
  pcout << "  Energy: " << energy << std::endl;

  if (energy_log_enabled && mpi_rank == 0)
    {
      energy_log_stream << time << "," << std::setprecision(16) << energy << "\n";
      energy_max = std::max(energy_max, energy);
      energy_min = std::min(energy_min, energy);
    }

  return energy;
}

void WaveEquation::enable_blowup_guard(const double blowup_factor_)
{
  blowup_guard_enabled = true;
  blowup_factor = blowup_factor_;
}

void WaveEquation::enable_energy_log(const std::string &csv_path)
{
  energy_log_enabled = true;

  const std::filesystem::path out_dir(output_dir);
  energy_log_path = (out_dir / csv_path).string();

  if (mpi_rank == 0)
    {
      std::filesystem::create_directories(out_dir);
      energy_log_stream.open(energy_log_path);
      energy_log_stream << "t,energy\n";
    }
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
  // M and K are constant in time here, so assemble them once. Time-dependent
  // coefficients or nonlinear terms would require reassembling each timestep.
  assemble_matrices();

  if (enable_output)
    output(); // Output initial state (t=0)
  const double e0 = compute_energy();
  if (energy_log_enabled)
    energy_initial = e0;

  while (time < T - 0.5 * delta_t) {
    time += delta_t;
    ++timestep_number;

    pcout << "Timestep " << timestep_number << " at t = " << time << std::endl;
    solve_timestep();
    const double energy_n = compute_energy();

    // Output periodically to save disk space if delta_t is very small
    if (enable_output && timestep_number % 2 == 0) {
      output();
    }

    if (blowup_guard_enabled && energy_n > blowup_factor * e0) {
      blowup_triggered = true;
      blowup_time_value = time;
      pcout << "[Blow-up guard] E^n = " << energy_n << " exceeded "
            << blowup_factor << " * E^0 at t = " << time
            << " -- stopping early." << std::endl;
      break;
    }
  }

  if (energy_log_enabled)
    {
      energy_log_stream.close();
      const double rel_variation = (energy_max - energy_min) / energy_initial;
      pcout << "===============================================" << std::endl;
      pcout << "Energy log written to " << energy_log_path << std::endl;
      pcout << "  E^0            = " << energy_initial << std::endl;
      pcout << "  max(E^n)       = " << energy_max << std::endl;
      pcout << "  min(E^n)       = " << energy_min << std::endl;
      pcout << "  (max-min)/E^0  = " << rel_variation << std::endl;
      pcout << "===============================================" << std::endl;
    }

  computing_timer.print_wall_time_statistics(MPI_COMM_WORLD);
}