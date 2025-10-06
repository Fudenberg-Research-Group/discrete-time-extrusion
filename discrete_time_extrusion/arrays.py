def make_site_array(xp,
                    type_list,
                    site_types,
                    value_dict,
                    at_ids=None,
                    number_of_replica=1,
                    **kwargs):
    
    assert len(type_list) == len(value_dict), ('Number of values (%d) incompatible with number of site types (%d)'
                                            % (len(value_dict), len(type_list)))
    
    prop_array = xp.zeros(len(site_types), dtype=xp.float64)
    
    for i, name in enumerate(type_list):
        prop_array[site_types == i] = value_dict[name]
        
    try:
        ids = xp.array(at_ids, dtype=xp.uint32)
        
        mask = xp.zeros(len(site_types), dtype=bool)
        mask[ids] = True
        
        prop_array[~mask] = 0
        
    except:
        pass
        
    return xp.tile(prop_array, number_of_replica)


def make_CTCF_arrays(xp,
                     type_list,
                     site_types,
                     left_positions,
                     right_positions,
                     CTCF_facestall,
                     CTCF_backstall,
                     velocity_multiplier,
                     **kwargs):
    
    stall_left_array = make_site_array(xp, type_list, site_types, CTCF_facestall,
                                       at_ids=left_positions, **kwargs)
    stall_right_array = make_site_array(xp, type_list, site_types, CTCF_facestall,
                                        at_ids=right_positions, **kwargs)
    
    stall_left_array += make_site_array(xp, type_list, site_types, CTCF_backstall,
                                        at_ids=right_positions, **kwargs)
    stall_right_array += make_site_array(xp, type_list, site_types, CTCF_backstall,
                                         at_ids=left_positions, **kwargs)
    
    stall_left_array = 1 - (1-stall_left_array) ** velocity_multiplier
    stall_right_array = 1 - (1-stall_right_array) ** velocity_multiplier

    return [stall_left_array, stall_right_array]


def make_CTCF_dynamic_arrays(xp,
                             type_list,
                             site_types,
                             CTCF_on_rate,
                             CTCF_off_rate,
                             sites_per_monomer,
                             velocity_multiplier,
                             **kwargs):
    
    on_rate_array = make_site_array(xp, type_list, site_types, CTCF_on_rate, **kwargs)
    off_rate_array = make_site_array(xp, type_list, site_types, CTCF_off_rate, **kwargs)
    
    birth_array = on_rate_array / (velocity_multiplier * sites_per_monomer)
    death_array = off_rate_array / (velocity_multiplier * sites_per_monomer)

    return [birth_array, death_array]
    

def make_LEF_arrays(xp,
                    type_list,
                    site_types,
					LEF_on_rate,
					LEF_off_rate,
					LEF_stalled_off_rate,
					LEF_diffusion_rate,
                    LEF_pause,
                    sites_per_monomer,
                    velocity_multiplier,
                    **kwargs):
    
    on_rate_array = make_site_array(xp, type_list, site_types, LEF_on_rate, **kwargs)
    off_rate_array = make_site_array(xp, type_list, site_types, LEF_off_rate, **kwargs)
    
    stalled_off_rate_array = make_site_array(xp, type_list, site_types, LEF_stalled_off_rate, **kwargs)
    diffusion_rate_array = make_site_array(xp, type_list, site_types, LEF_diffusion_rate, **kwargs)

    birth_array = on_rate_array / (velocity_multiplier * sites_per_monomer)
    death_array = off_rate_array / (velocity_multiplier * sites_per_monomer)
    
    stalled_death_array = stalled_off_rate_array / (velocity_multiplier * sites_per_monomer)
    diffusion_array = diffusion_rate_array / (velocity_multiplier * sites_per_monomer)
    
    pause_array = make_site_array(xp, type_list, site_types, LEF_pause, **kwargs)

    return [birth_array, death_array, stalled_death_array, diffusion_array, pause_array]


def make_LEF_transition_dict(xp,
                             type_list,
                             site_types,
                             LEF_states,
                             LEF_transition_rates,
                             sites_per_monomer,
                             velocity_multiplier,
                             **kwargs):
    
    transition_dict = {}
    
    transition_dict["LEF_states"] = LEF_states
    transition_dict["LEF_transitions"] = {}

    for ids, LEF_rate in LEF_transition_rates.items():
        rate_array = make_site_array(xp, type_list, site_types, LEF_rate, **kwargs)
        transition_dict["LEF_transitions"][ids] = rate_array / (velocity_multiplier * sites_per_monomer)

    return transition_dict
