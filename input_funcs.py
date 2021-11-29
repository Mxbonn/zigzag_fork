import importlib.machinery

from msg import MemoryNode, MemorySchemeNode


class InputSettings:

    def __init__(self, results_path, results_filename, layer_filename, layer_number, layer_parallel_processing, precision,
                 mac_array_info, mac_array_stall, mem_hierarchy_single_simulation, mem_scheme_parallel_processing,
                 mem_scheme_single, fixed_spatial_unrolling, spatial_unrolling_single, flooring_single,
                 fixed_temporal_mapping, temporal_mapping_single, tmg_search_method, tmg_search_space, temporal_mapping_multiprocessing,
                 drc_enabled, PE_RF_size_threshold, PE_RF_depth, CHIP_depth, max_area, utilization_rate_area,
                 memory_hierarchy_ratio, mem_pool, banking, L1_size, L2_size, unrolling_size_list, unrolling_scheme_list,
                 unrolling_scheme_list_text, memory_scheme_hint, mh_name, spatial_utilization_threshold, spatial_unrolling_mode,
                 stationary_optimization_enable, su_parallel_processing, arch_search_result_saving, su_search_result_saving,
                 tm_search_result_saving, result_print_mode, im2col_enable_all, im2col_enable_pw, memory_unroll_fully_flexible,
                 result_print_type, save_results_on_the_fly, max_nb_lpf_layer):
        self.results_path = results_path
        self.results_filename = results_filename
        self.layer_filename = layer_filename
        self.layer_number = layer_number
        self.layer_parallel_processing = layer_parallel_processing
        self.precision = precision
        self.mac_array_info = mac_array_info
        self.mac_array_stall = mac_array_stall
        self.energy_over_utilization = True
        self.mem_hierarchy_single_simulation = mem_hierarchy_single_simulation
        self.mem_scheme_parallel_processing = mem_scheme_parallel_processing
        self.mem_scheme_single = mem_scheme_single
        self.fixed_spatial_unrolling = fixed_spatial_unrolling
        self.spatial_unrolling_single = spatial_unrolling_single
        self.flooring_single = flooring_single
        self.fixed_temporal_mapping = fixed_temporal_mapping
        self.temporal_mapping_single = temporal_mapping_single
        self.tmg_search_method = tmg_search_method
        self.tmg_search_space = tmg_search_space
        self.temporal_mapping_multiprocessing = temporal_mapping_multiprocessing
        self.drc_enabled = drc_enabled
        self.prune_PE_RF = True
        self.mem_hierarchy_iterative_search = False
        self.unrolling_size_list = unrolling_size_list
        self.unrolling_scheme_list = unrolling_scheme_list
        self.unrolling_scheme_list_text = unrolling_scheme_list_text
        self.PE_RF_size_threshold = PE_RF_size_threshold
        self.PE_RF_depth = PE_RF_depth
        self.CHIP_depth = CHIP_depth
        self.utilization_optimizer_pruning = False
        self.max_area = max_area
        self.utilization_rate_area = utilization_rate_area
        self.memory_hierarchy_ratio = memory_hierarchy_ratio
        self.mem_pool = mem_pool
        self.L1_size = L1_size
        self.L2_size = L2_size
        self.memory_scheme_hint = memory_scheme_hint
        self.mh_name = mh_name
        self.banking = banking
        self.spatial_utilization_threshold = spatial_utilization_threshold
        self.spatial_unrolling_mode = spatial_unrolling_mode
        self.stationary_optimization_enable = stationary_optimization_enable
        self.su_parallel_processing = su_parallel_processing
        self.arch_search_result_saving = arch_search_result_saving
        self.su_search_result_saving = su_search_result_saving
        self.tm_search_result_saving = tm_search_result_saving
        self.result_print_mode = result_print_mode
        self.im2col_enable_all = im2col_enable_all
        self.im2col_enable_pw = im2col_enable_pw
        # TODO im2col_top_mem_level
        self.im2col_top_mem_level = 100
        self.memory_unroll_fully_flexible = memory_unroll_fully_flexible
        self.result_print_type = result_print_type
        self.save_results_on_the_fly = save_results_on_the_fly
        self.max_nb_lpf_layer = max_nb_lpf_layer


def get_input_settings(settings_cfg, mapping_cfg, memory_pool_cfg, architecture_cfg):
    if settings_cfg['result_print_mode'] not in ['concise', 'complete']:
        raise ValueError('result_print_mode is not correctly set. Please check the setting file.')

    if settings_cfg['result_print_type'] not in ['xml', 'yaml']:
        raise ValueError('result_print_type is not correctly set. Please check the setting file.')

    tm_fixed_flag = settings_cfg['fixed_temporal_mapping']
    sm_fixed_flag = settings_cfg['fixed_spatial_unrolling']
    arch_fixed_flag = settings_cfg['fixed_architecture']
    memory_pool = []
    for m in memory_pool_cfg:
        mt = memory_pool_cfg[m]['mem_type']
        mt_tmp = 0
        if mt == 'dual_port_double_buffered':
            mt_tmp = 3
        elif mt == 'dual_port_single_buffered':
            mt_tmp = 2
        elif mt == 'single_port_double_buffered':
            mt_tmp = 1
        if mt_tmp == 0:
            raise ValueError("In memory pool, some memory's memory type is not correctly defined.")
        mbw = []
        for mb in memory_pool_cfg[m]['mem_bw']:
            mbw.append([mb, mb])
        try:
            mem_utilization_rate = memory_pool_cfg[m]['utilization_rate']
        except:
            mem_utilization_rate = 0.7
        m_tmp = {
            'name': m,
            'size_bit': memory_pool_cfg[m]['size_bit'],
            'mem_bw': mbw,
            'area': memory_pool_cfg[m]['area'],
            'utilization_rate': mem_utilization_rate,
            'mem_type': mt_tmp,
            'cost': [list(a) for a in zip(memory_pool_cfg[m]['cost']['read_word'], memory_pool_cfg[m]['cost']['write_word'])],
            'unroll': 1,
            'mem_fifo': False
        }
        memory_pool.append(m_tmp)
    try:
        memory_unroll_fully_flexible = architecture_cfg['memory_unroll_fully_flexible']
    except:
        memory_unroll_fully_flexible = False
    L1_size = architecture_cfg['L1_size']
    L2_size = architecture_cfg['L2_size']
    banking = architecture_cfg['banking']
    area_max_arch = architecture_cfg['area_max']
    area_utilization_arch = architecture_cfg['area_utilization']
    mem_ratio = architecture_cfg['mem_ratio']
    PE_depth = architecture_cfg['PE_memory_depth']
    CHIP_depth = architecture_cfg['CHIP_memory_depth']
    PE_RF_size_threshold = architecture_cfg['PE_threshold']
    mac_array_info = {}
    mac_array_stall = {}
    mac_array_info['array_size'] = [architecture_cfg['PE_array']['Col'], architecture_cfg['PE_array']['Row']]
    memory_scheme_hint = MemorySchemeNode([])
    mh_name = {'W': [], 'I': [], 'O': []}

    if arch_fixed_flag:
        for m in architecture_cfg['memory_hierarchy']:
            m_tmp = [x for x in memory_pool if x['name'] == architecture_cfg['memory_hierarchy'][m]['memory_instance']]
            if not m_tmp:
                raise Exception("Memory instance " + str(m) + " in hierarchy is not found in memory pool")
            m_tmp = m_tmp[0]
            m_tmp = MemoryNode(m_tmp, (), 0, 1, m)
            m_tmp.memory_level['unroll'] = architecture_cfg['memory_hierarchy'][m]['memory_unroll']
            m_tmp.memory_level['nbanks'] = None
            m_tmp.operand = tuple(architecture_cfg['memory_hierarchy'][m]['operand_stored'])
            memory_scheme_hint.memory_scheme.add(m_tmp)
            for operand in ['W', 'I', 'O']:
                if operand in m_tmp.operand:
                    mh_name[operand].append(tuple([m, m_tmp.memory_level['size_bit']]))
        for operand in ['W', 'I', 'O']:
            mh_name[operand].sort(key=lambda tup: tup[1])
            mh_name[operand] = [x[0] for x in mh_name[operand]]
    else:
        if architecture_cfg['memory_hint']:
            for m in architecture_cfg['memory_hint']:
                m_tmp = [x for x in memory_pool if x['name'] == architecture_cfg['memory_hint'][m]['memory_instance']]
                if not m_tmp:
                    raise Exception("Memory instance " + str(m) + " in hint is not found in memory pool")
                m_tmp = m_tmp[0]
                memory_pool.remove(m_tmp)
                m_tmp = MemoryNode(m_tmp, (), 0, 1)
                m_tmp.memory_level['unroll'] = architecture_cfg['memory_hint'][m]['memory_unroll']
                m_tmp.memory_level['nbanks'] = 1
                m_tmp.operand = tuple(architecture_cfg['memory_hint'][m]['operand_stored'])

                memory_scheme_hint.memory_scheme.add(m_tmp)

    precision = {'W': architecture_cfg['precision']['W'], 'I': architecture_cfg['precision']['I'], 'O': architecture_cfg['precision']['O_partial'],
                 'O_final': architecture_cfg['precision']['O_final']}
    mac_array_info['single_mac_energy'] = architecture_cfg['single_mac_energy_active']
    mac_array_info['idle_mac_energy'] = architecture_cfg['single_mac_energy_idle']
    p_aux = [precision['W'], precision['I']]
    mac_array_info['precision'] = max(p_aux)
    mac_array_info['headroom'] = precision['O'] - precision['O_final']

    uwe = []
    for i in range(0, 10):
        uwe.append(0)
    mac_array_info['unit_wire_energy'] = uwe
    mac_array_stall['systolic'] = architecture_cfg['mac_array_stall']['systolic']
    tm_fixed = {'W': [], 'I': [], 'O': []}
    sm_fixed = {'W': [], 'I': [], 'O': []}
    flooring_fixed = {'W': [], 'I': [], 'O': []}
    unrolling_scheme_list = []
    unrolling_size_list = []
    i2a = {'B': 7, 'K': 6, 'C': 5, 'OY': 4, 'OX': 3, 'FY': 2, 'FX': 1}
    unrolling_scheme_list_text = []
    if tm_fixed_flag:
        for op in mapping_cfg['temporal_mapping_fixed']:
            if op == 'weight':
                operand = 'W'
            elif op == 'input':
                operand = 'I'
            elif op == 'output':
                operand = 'O'
            tm_fixed[operand] = [[] for x in mapping_cfg['temporal_mapping_fixed'][op]]
            for ii, lev in enumerate(mapping_cfg['temporal_mapping_fixed'][op]):
                index_lev = mh_name[operand].index(lev)
                for pf in mapping_cfg['temporal_mapping_fixed'][op][lev]:
                    tm_fixed[operand][index_lev].append(tuple([i2a[pf[0]], pf[1]]))
    if sm_fixed_flag:
        for op in mapping_cfg['spatial_mapping_fixed']:
            if op == 'weight':
                operand = 'W'
            elif op == 'input':
                operand = 'I'
            elif op == 'output':
                operand = 'O'
            sm_fixed[operand] = [[] for x in mapping_cfg['spatial_mapping_fixed'][op]]
            flooring_fixed[operand] = [[] for x in mapping_cfg['spatial_mapping_fixed'][op]]
            for lev in mapping_cfg['spatial_mapping_fixed'][op]:
                ii_lev = 0
                if lev == 'MAC':
                    ii_lev = 0
                else:
                    ii_lev = lev + 1
                flooring_fixed[operand][ii_lev] = [[] for d in mapping_cfg['spatial_mapping_fixed'][op][lev]]
                for dim in mapping_cfg['spatial_mapping_fixed'][op][lev]:
                    ii_dim = 0
                    if dim == 'Col':
                        ii_dim = 0
                    elif dim == 'Row':
                        ii_dim = 1
                    for pf in mapping_cfg['spatial_mapping_fixed'][op][lev][dim]:
                        sm_fixed[operand][ii_lev].append(tuple([i2a[pf[0]], pf[1]]))
                        flooring_fixed[operand][ii_lev][ii_dim].append(i2a[pf[0]])
    else:
        unrolling_scheme_list_text = mapping_cfg['spatial_mapping_list']
        for us in mapping_cfg['spatial_mapping_list']:
            unrolling_scheme_list.append([])
            unrolling_scheme_list[-1] = [[] for x in us]
            unrolling_size_list.append([])
            unrolling_size_list[-1] = [[] for x in us]
            for dim in us:
                ii_dim = 0
                dimx = next(iter(dim))
                if dimx == 'Col':
                    ii_dim = 0
                elif dimx == 'Row':
                    ii_dim = 1
                for pf in dim[dimx]:
                    pf_type = list(pf.split('_'))[0]
                    unrolling_scheme_list[-1][ii_dim].append(i2a[pf_type])
                    try:
                        pf_size = list(pf.split('_'))[1]
                        unrolling_size_list[-1][ii_dim].append(int(pf_size))
                    except:
                        pf_size = None
                        unrolling_size_list[-1][ii_dim].append(pf_size)

    if settings_cfg['temporal_mapping_search_method'] == 'exhaustive':
        tmg_search_method = 1
        stationary_optimization_enable = False
        data_reuse_threshold = 0
    elif settings_cfg['temporal_mapping_search_method'] == 'iterative':
        tmg_search_method = 0
        stationary_optimization_enable = True
        data_reuse_threshold = 1
    elif settings_cfg['temporal_mapping_search_method'] == 'heuristic_v1':
        tmg_search_method = 1
        stationary_optimization_enable = True
        data_reuse_threshold = 0
    elif settings_cfg['temporal_mapping_search_method'] == 'heuristic_v2':
        tmg_search_method = 1
        stationary_optimization_enable = True
        data_reuse_threshold = 1
    elif settings_cfg['temporal_mapping_search_method'] == 'loma':
        tmg_search_method = 2
        stationary_optimization_enable = None
        data_reuse_threshold = None
    else:
        raise ValueError('temporal_mapping_search_method is not correctly set. Please check the setting file.')

    # Temporal mapping search space: even or uneven
    try:
        if settings_cfg['temporal_mapping_search_space'] == 'even':
            if tmg_search_method == 2:  # Only allow even mapping when doing LOMA
                tmg_search_space = 'even'
            else:
                raise ValueError('temporal_mapping_search_space = even is only allowed for LOMA search.')
        elif settings_cfg['temporal_mapping_search_space'] == 'uneven':
            tmg_search_space = 'uneven'
        else:
            raise ValueError('temporal_mapping_search_space is not correctly set. Please check the setting file.')
    except:
        print("Temporal mapping search space not defined, generating uneven mappings.")
        tmg_search_space = 'uneven'

    sumode = ['exhaustive', 'heuristic_v1', 'heuristic_v2', 'hint_driven', 'greedy_mapping_with_hint', 'greedy_mapping_without_hint']
    if not settings_cfg['fixed_spatial_unrolling']:
        sumx = sumode.index(settings_cfg['spatial_unrolling_search_method'])
    else:
        sumx = 0

    if type(settings_cfg['layer_indices']) is list:
        layer_indices = settings_cfg['layer_indices']
    else:
        NN = importlib.machinery.SourceFileLoader('%s' % (settings_cfg['layer_filename']), '%s.py' % (settings_cfg['layer_filename'])).load_module()
        layer_indices = [kk for kk in NN.layer_info.keys()]

    try:
        save_results_on_the_fly = settings_cfg['save_results_on_the_fly']
    except:
        save_results_on_the_fly = False
    try:
        max_nb_lpf_layer = settings_cfg['max_nb_lpf_layer']
    except:
        max_nb_lpf_layer = 20

    input_settings_cfg = InputSettings(settings_cfg['result_path'], settings_cfg['result_filename'], settings_cfg['layer_filename'],
                                       layer_indices, settings_cfg['layer_multiprocessing'], precision,
                                       mac_array_info, mac_array_stall, settings_cfg['fixed_architecture'],
                                       settings_cfg['architecture_search_multiprocessing'], memory_scheme_hint,
                                       settings_cfg['fixed_spatial_unrolling'], sm_fixed, flooring_fixed,
                                       settings_cfg['fixed_temporal_mapping'], tm_fixed, tmg_search_method,
                                       tmg_search_space, settings_cfg['temporal_mapping_multiprocessing'],
                                       data_reuse_threshold, PE_RF_size_threshold, PE_depth,
                                       CHIP_depth, area_max_arch, area_utilization_arch,
                                       mem_ratio, memory_pool, banking, L1_size, L2_size, unrolling_size_list,
                                       unrolling_scheme_list, unrolling_scheme_list_text, memory_scheme_hint, mh_name,
                                       settings_cfg['spatial_utilization_threshold'], sumx, stationary_optimization_enable,
                                       settings_cfg['spatial_unrolling_multiprocessing'], settings_cfg['save_all_architecture_result'],
                                       settings_cfg['save_all_spatial_unrolling_result'], settings_cfg['save_all_temporal_mapping_result'],
                                       settings_cfg['result_print_mode'], settings_cfg['im2col_enable_for_all_layers'],
                                       settings_cfg['im2col_enable_for_pointwise_layers'], memory_unroll_fully_flexible,
                                       settings_cfg['result_print_type'], save_results_on_the_fly, max_nb_lpf_layer)

    return input_settings_cfg


class layer_spec1(object):
    def __init__(self):
        self.layer_info = {}


def get_layer_spec(input_settings, model=None):
    """
    Function that gets the layer_spec according from the input_settings
    If a Keras model is provided, it will update the layer spec accordingly


    Arguments
    =========

    - input_settings: The input settings to get the layer_spec file location

    - model: A keras model that constitutes of a number of Conv2D layers

    """
    if input_settings:
        layer_filename = input_settings.layer_filename
        layer_spec = importlib.machinery.SourceFileLoader('%s' % (layer_filename), '%s.py' % (layer_filename)).load_module()
        layer_numbers = input_settings.layer_number
    else:
        layer_spec = layer_spec1()

    if model is not None:
        layer_numbers = update_layer_spec(layer_spec, model)

    for layer_number, specs in layer_spec.layer_info.items():
        if layer_number in layer_numbers:  # Only care about layers we have to process
            G = specs.get('G', 1)
            C = specs['C']
            K = specs['K']

            if G != 1:
                div_C, mod_C = divmod(C, G)
                div_K, mod_K = divmod(K, G)

                assert (mod_C == 0 and mod_K == 0), "C and/or K not divisible by number of groups for layer %d" % layer_number
                layer_spec.layer_info[layer_number]['C'] = div_C
                layer_spec.layer_info[layer_number]['K'] = div_K

                print("Grouped convolution detected for %s Layer %d. Terminal prints will show total energy of all groups combined."
                      % (input_settings.layer_filename.split('/')[-1], layer_number))
    print()
    return layer_spec, layer_numbers


def update_layer_spec(layer_spec, model):
    """
    Function that changes the layer_spec according to a keras model.

    Arguments
    =========
    - layer_spec: The layer_spec module that will be updated

    - input_settings: The input settings, needed to update the layer_number variable

    - model: A keras model that constitutes of a number of Conv2D layers

    """
    import keras

    # Clear any entries present in layer_spec
    layer_spec.layer_info = {}
    layer_numbers = []
    layer_ii = 0
    # Iterate through model layers
    for layer_idx, layer in enumerate(model.layers):
        layer_number = layer_idx + 1
        print(layer_idx, type(layer))

        # Get the specs for this layer
        if isinstance(layer, keras.layers.Conv1D) or \
                isinstance(layer, keras.layers.Conv2D) or \
                isinstance(layer, keras.layers.Conv3D) or \
                isinstance(layer, keras.layers.SeparableConv1D) or \
                isinstance(layer, keras.layers.SeparableConv2D) or \
                isinstance(layer, keras.layers.DepthwiseConv2D) or \
                isinstance(layer, keras.layers.Dense):
            layer_ii += 1
            b = layer.input_shape[0]
            if b is None:
                b = 1
            if isinstance(layer, keras.layers.SeparableConv1D) or \
                    isinstance(layer, keras.layers.SeparableConv2D):

                # manually split a SeparableConv into 2 layers: depthwise & pointwise
                c = layer.input_shape[3]
                ox = layer.output_shape[1]
                oy = layer.output_shape[2]
                k = layer.input_shape[3] * layer.depth_multiplier
                fx = layer.kernel_size[0]
                fy = layer.kernel_size[1]
                sx = layer.strides[0]
                sy = layer.strides[1]
                sfx = layer.dilation_rate[0]
                sfy = layer.dilation_rate[1]
                px = 0
                py = 0
                g = layer.input_shape[3]

                # Update the layer_spec variable
                layer_spec.layer_info[layer_ii] = {
                    'B': b,
                    'K': k,
                    'C': c,
                    'OY': oy,
                    'OX': ox,
                    'FY': fy,
                    'FX': fx,
                    'SY': sy,
                    'SX': sx,
                    'SFY': sfy,
                    'SFX': sfx,
                    'PY': py,
                    'PX': px,
                    'G': g
                }

                # Add this layer number to layer_numbers
                layer_numbers.append(layer_ii)
                layer_ii += 1

                c = layer.output_shape[3] * layer.depth_multiplier
                ox = layer.output_shape[1]
                oy = layer.output_shape[2]
                k = layer.output_shape[3]
                fx = 1
                fy = 1
                sx = 1
                sy = 1
                sfx = 1
                sfy = 1
                px = 0
                py = 0
                g = 1


            elif isinstance(layer, keras.layers.DepthwiseConv2D):
                c = layer.input_shape[3]
                ox = layer.output_shape[1]
                oy = layer.output_shape[2]
                k = layer.output_shape[3]
                fx = layer.kernel_size[0]
                fy = layer.kernel_size[1]
                sx = layer.strides[0]
                sy = layer.strides[1]
                sfx = layer.dilation_rate[0]
                sfy = layer.dilation_rate[1]
                px = 0
                py = 0
                g = c
                if c != k:
                    raise ("ERROR: C!=K")

            elif isinstance(layer, keras.layers.Dense):
                # fully-connected layer
                c = layer.input_shape[1]
                ox = 1
                oy = 1
                k = layer.output_shape[1]
                fx = 1
                fy = 1
                sx = 1
                sy = 1
                sfx = 1
                sfy = 1
                px = 0
                py = 0
                g = 1

            else:
                c = layer.input_shape[3]
                ox = layer.output_shape[1]
                oy = layer.output_shape[2]
                k = layer.output_shape[3]
                fx = layer.kernel_size[0]
                fy = layer.kernel_size[1]
                sx = layer.strides[0]
                sy = layer.strides[1]
                sfx = layer.dilation_rate[0]
                sfy = layer.dilation_rate[1]
                px = 0
                py = 0
                g = 1

            # Update the layer_spec variable
            layer_spec.layer_info[layer_ii] = {
                'B': b,
                'K': k,
                'C': c,
                'OY': oy,
                'OX': ox,
                'FY': fy,
                'FX': fx,
                'SY': sy,
                'SX': sx,
                'SFY': sfy,
                'SFX': sfx,
                'PY': py,
                'PX': px,
                'G': g
            }

            # Add this layer number to layer_numbers
            layer_numbers.append(layer_ii)

    return layer_numbers
