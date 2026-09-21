// Discrete Fresh Concrete contact law for DEM-Engine.
// Ported from FreshConcreteContact.cpp in chrono-mechanics. Units are supplied
// consistently by the caller (the L-box uses mm, s, and its derived force unit).

if (overlapDepth > 0.f) {
    // Keep model-specific utilities local to the generated contact kernel. CUDA
    // lambdas inherit the execution space of their enclosing function, so no
    // separately injected prerequisite source is needed.
    const auto safe_basis_y = [](const float3 n) {
        const float3 ref = fabsf(n.y) < 0.9f ? make_float3(0.f, 1.f, 0.f)
                                              : make_float3(0.f, 0.f, 1.f);
        return normalize(cross(normalize(cross(n, ref)), n));
    };
    const auto floc_rhs = [](float lambda_value, float shear_rate,
                             float beta_value, float exponent, float critical_time) {
        lambda_value = fmaxf(lambda_value, 1.e-8f);
        return 1.f / (critical_time * powf(lambda_value, exponent))
               - beta_value * shear_rate * lambda_value;
    };

    const float h_mat = mortar_layer_mat[bodyAMatType];
    const float ENm = ENm_mat[bodyAMatType];
    const float ENa = ENa_mat[bodyAMatType];
    const float alpha = alpha_mat[bodyAMatType];
    const float beta = beta_mat[bodyAMatType];
    const float np = np_mat[bodyAMatType];
    const float sgm_tau0 = sgmTau0_mat[bodyAMatType];
    const float eta_inf = eta_inf_mat[bodyAMatType];
    const float eta0 = kappa0_mat[bodyAMatType] * eta_inf;
    const float deps0 = sgm_tau0 / fmaxf(eta0, 1.e-20f);

    float lambda = contact_info_lambda;
    if (lambda <= 0.f)
        lambda = lambda_init_mat[bodyAMatType];

    const float3 normal = -B2A;
    const float3 tangent_m = safe_basis_y(normal);
    const float3 tangentLong = normalize(cross(normal, tangent_m));

    float h = h_mat;
    float r1 = ARadius;
    float r2 = BRadius;
    float rmin, rmax, delta_rmin, l0;
    const float delta = overlapDepth;

    if (ContactType == deme::SPHERE_SPHERE_CONTACT) {
        l0 = r1 + r2 - h;
        rmin = fminf(r1, r2);
        rmax = fmaxf(r1, r2);
    } else {
        // The wall owns half of the mortar film; its artificial curvature is
        // retained from the reference implementation.
        h *= .5f;
        r2 = ContactType == deme::SPHERE_MESH_CONTACT ? 5.f : 6.f;
        l0 = r1 + r2 - h;
        rmin = r1;
        rmax = fmaxf(r1, r2);
    }

    const float lij = fmaxf(fabsf(r1 + r2 - delta), 1.e-7f);
    if (ContactType == deme::SPHERE_SPHERE_CONTACT)
        delta_rmin = delta * .5f * (2.f*rmax - delta) / lij;
    else
        delta_rmin = delta;

    const float ai = fabsf(rmin - delta_rmin);
    const float contact_area = deme::PI * fmaxf(rmin*rmin - ai*ai, 0.f);

    float3 rot_a = cross(ARotVel, locCPA);
    float3 rot_b = cross(BRotVel, locCPB);
    applyOriQToVector3<float, deme::oriQ_t>(rot_a.x, rot_a.y, rot_a.z,
                                            AOriQ.w, AOriQ.x, AOriQ.y, AOriQ.z);
    applyOriQToVector3<float, deme::oriQ_t>(rot_b.x, rot_b.y, rot_b.z,
                                            BOriQ.w, BOriQ.x, BOriQ.y, BOriQ.z);
    const float3 relative_velocity = (BLinVel + rot_b) - (ALinVel + rot_a);
    const float inv_lij = 1.f / lij;
    const float vdeps_n = dot(relative_velocity, normal) * inv_lij;
    const float vdeps_m = dot(relative_velocity, tangent_m) * inv_lij;
    const float rateLong = dot(relative_velocity, tangentLong) * inv_lij;
    const float deps_m = vdeps_m * ts;
    const float incrementLong = rateLong * ts;

    const float eps_a = logf(fmaxf(1.f - h / l0, 1.e-8f));
    const float eps_n = logf(fmaxf(lij / l0, 1.e-8f));
    float eps_m = contact_info_strain_y + deps_m;
    float strainLong = contact_info_strain_z + incrementLong;

    contact_info_strain_x = eps_n;
    contact_info_strain_y = eps_m;
    contact_info_strain_z = strainLong;
    contact_info_step_time = time;

    float normal_stress;
    if (eps_n < 0.f) {
        if (eps_n >= eps_a) {
            normal_stress = ENm * eps_n;
            eps_m = strainLong = 0.f;
            contact_info_strain_y = 0.f;
            contact_info_strain_z = 0.f;
        } else {
            normal_stress = ENm * eps_a + ENa * (eps_n - eps_a);
        }
    } else {
        normal_stress = fminf(ENm * eps_n, sgmTmax_mat[bodyAMatType] * (1.f + lambda));
    }

    const float equivalent_rate = sqrtf(beta*vdeps_n*vdeps_n
                                       + vdeps_m*vdeps_m + rateLong*rateLong);
    const float floc_beta = flocbeta_mat[bodyAMatType];
    const float floc_exponent = flocm_mat[bodyAMatType];
    const float floc_critical_time = flocTcr_mat[bodyAMatType];
    const float k1 = floc_rhs(lambda, equivalent_rate, floc_beta,
                              floc_exponent, floc_critical_time);
    const float k2 = floc_rhs(lambda + .5f*ts*k1, equivalent_rate, floc_beta,
                              floc_exponent, floc_critical_time);
    const float k3 = floc_rhs(lambda + .5f*ts*k2, equivalent_rate, floc_beta,
                              floc_exponent, floc_critical_time);
    const float k4 = floc_rhs(lambda + ts*k3, equivalent_rate, floc_beta,
                              floc_exponent, floc_critical_time);
    lambda = fmaxf(lambda + ts*(k1 + 2.f*k2 + 2.f*k3 + k4)/6.f, 0.f);
    contact_info_lambda = lambda;

    float viscosity = eta0;
    if (equivalent_rate > deps0)
        viscosity = eta_inf * powf(equivalent_rate, np - 1.f)
                    + sgm_tau0 * (1.f + lambda) / equivalent_rate;

    const float viscous_normal_stress = beta * viscosity * vdeps_n;
    const float viscous_tangent_m_stress = viscosity * vdeps_m;
    const float viscousLongStress = viscosity * rateLong;
    force = contact_area * ((normal_stress + viscous_normal_stress) * normal
                            + viscous_tangent_m_stress * tangent_m
                            + viscousLongStress * tangentLong);
} else {
    force = make_float3(0.f, 0.f, 0.f);
    contact_info_step_time = 0.f;
    contact_info_lambda = 0.f;
    contact_info_strain_x = 0.f;
    contact_info_strain_y = 0.f;
    contact_info_strain_z = 0.f;
}
