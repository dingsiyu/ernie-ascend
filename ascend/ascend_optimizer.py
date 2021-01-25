import paddle.fluid.framework as framework
from paddle.fluid.optimizer import Optimizer
import paddle.fluid.core as core
import numpy as np
from ascend import ascend_parser

class AscendIRParser(object):
    def __init__(self):
        self.graph_idx = 0
        
    def _construct_input_map(self, input_varlist):
        ret_map = {}
        ge_in_operator = []
        for id, var in enumerate(input_varlist):
            if var.is_data: # fluid.data
                print("_construct_input_map for %d input var[%s]" % (id, var.name))
                ge_input = core.GEOperatorFactory.create_operator(var.name, "Data").set_attr_int32("index", id)
                ret_map[var.name] = ge_input
                ge_in_operator.append(ge_input)
            else: # param
                print("_construct_input_map for %d param var[%s]" % (id, var.name))
                ge_input = core.GEOperatorFactory.create_operator(var.name, "Variable")
                ge_input.update_output_desc("y", core.GETensorDesc(core.GEShape(var.shape), core.GEFormat.FORMAT_ND, core.GEDataType.DT_FLOAT))
                ret_map[var.name] = ge_input
                # ge_in_operator.append(ge_input)
        return ge_in_operator, ret_map

    def parse_op(self, op):
        if op.type in ascend_parser.registerd_op:
            print("op[%s] has been registered" % (op.type))
            op_parser = self.parser_factory.create_parse(ascend_parser.registerd_op[op.type])
            op_parser.apply(op)
        else:
            print("op[%s] has not been registered, parse failed..." % (op.type))
            
    def _parse_program(self, graph_name, program, input_varlist=[], fetch_list=[]):
        begin_graph_idx = self.graph_idx
        subgraphs = []
        ge_in_operator = []
        ge_out_operator = []
        self.var2geop = {}

        block = program.global_block()
        if len(block.ops) == 0:
            print("there is no ops in program")
            return []

        graph = core.GEGraph(graph_name)
        print("begin parse %s" % (graph_name)) 

        ge_in_operator, self.var2geop = self._construct_input_map(input_varlist)
        # for k in self.var2geop:
        #     graph.add_op(self.var2geop[k])
        self.parser_factory = ascend_parser.AscendParserFactory(graph, self.var2geop)
        
        for i, curop in list(enumerate(block.ops)):
            self.parse_op(curop)
           
        for e in fetch_list:
            name = e
            if not isinstance(e, str):
                name = e.name
            ge_out_operator.append(self.var2geop[name])
        # hack back prop vars
        # ge_in_operator.append(self.var2geop["reduce_sum_0.tmp_0@GRAD"])
        #if begin_graph_idx > 0:
            #ge_in_operator.append(self.var2geop["scale_expand"])
            #ge_in_operator.append(self.var2geop["bias_expand"])
            #ge_in_operator.append(self.var2geop["sqrt_0.tmp_0"])
            #ge_in_operator.append(self.var2geop["x_var"])
            #ge_in_operator.append(self.var2geop["index_var"])
            #ge_in_operator.append(self.var2geop["updates_var"])

            #ge_out_operator[-1] = self.var2geop["fc_0.w_0@GRAD"] 
            #ge_out_operator[-1] = self.var2geop["sqrt_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["log_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["tanh_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["pow_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["fc_0.tmp_1@GRAD"] 
            #ge_out_operator[-1] = self.var2geop["mean_0.tmp_0@GRAD"]

            #ge_out_operator[-1] = self.var2geop["elementwise_mul_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["elementwise_add_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["elementwise_div_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["softmax_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["reshape2_0.tmp_0@GRAD"]
            #ge_out_operator[-2] = self.var2geop["gather_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["transpose_0.tmp_0@GRAD"]
            #ge_out_operator[-1] = self.var2geop["layer_norm_0.tmp_2"]

        # ge_out_operator[3] = self.var2geop["softmax_with_cross_entropy_0.tmp_1@GRAD"]
        print("ge_out_operator: ", ge_out_operator)
        if len(ge_in_operator) == 0: # ge graph must have at least one input and one output
            params_list = ["learning_rate_0",
                           "pre_encoder_layer_norm_bias",
                           "pre_encoder_layer_norm_scale",
                           "task_embedding",
                           "sent_embedding",
                           "pos_embedding",
                           "word_embedding"
                           ]
            for cnt in range(3):
                par_list = ["post_ffn_layer_norm_bias",
                            "post_ffn_layer_norm_scale",
                            "ffn_fc_1.b_0",
                            "ffn_fc_1.w_0",
                            "ffn_fc_0.b_0",
                            "ffn_fc_0.w_0",
                            "post_att_layer_norm_bias",
                            "post_att_layer_norm_scale",
                            "multi_head_att_output_fc.b_0",
                            "multi_head_att_output_fc.w_0",
                            "multi_head_att_value_fc.b_0",
                            "multi_head_att_value_fc.w_0",
                            "multi_head_att_key_fc.b_0",
                            "multi_head_att_key_fc.w_0",
                            "multi_head_att_query_fc.b_0",
                            "multi_head_att_query_fc.w_0"
                            ]
                for par in par_list:
                    par_name = "_".join(["encoder_layer", str(cnt), par])
                    params_list.append(par_name)
            
            params_list.extend(['mask_lm_out_fc.b_0',
                                'mask_lm_trans_layer_norm_bias',
                                'mask_lm_trans_layer_norm_scale',
                                'mask_lm_trans_fc.b_0',
                                'mask_lm_trans_fc.w_0',
                                'pooled_fc.b_0',
                                'pooled_fc.w_0'])

            for param in params_list:
                ge_in_operator.append(self.var2geop[param])
             
            #ge_in_operator.append(self.var2geop["learning_rate_0"])  
            #ge_in_operator.append(self.var2geop["fc_0.w_0"])
            #ge_in_operator.append(self.var2geop["fc_1.w_0"])
            #ge_in_operator.append(self.var2geop["layer_norm_0.w_0"])
            #ge_in_operator.append(self.var2geop["layer_norm_0.b_0"])
            
            name = "truncated_normal_input_tensor"
            append_var_list = []
            for var_name in self.var2geop.keys():
                if name in var_name:
                    append_var_list.append(var_name)
            if len(append_var_list) > 0:
                for var in append_var_list:
                    ge_in_operator.append(self.var2geop[var])
                
            #ge_in_operator.append(self.var2geop["1"])
            #ge_in_operator.append(self.var2geop["2"])
            #ge_in_operator.append(self.var2geop["3"])
            #ge_in_operator.append(self.var2geop["4"])
            #ge_in_operator.append(self.var2geop["5"])

            #ge_in_operator.append(self.var2geop["learning_rate_0"]) 
            ge_out_operator.append(self.var2geop["learning_rate_0"])
            #ge_out_operator.append(self.var2geop["fc_1.w_0"])
            #ge_out_operator.append(self.var2geop["layer_norm_0.w_0"])
            #ge_out_operator.append(self.var2geop["layer_norm_0.b_0"])
            #ge_out_operator.append(self.var2geop["fc_1.w_0"])
            #ge_out_operator.append(self.var2geop["learning_rate_0"])
            # graph.add_op(c1)
            
        graph.set_inputs(ge_in_operator).set_outputs(ge_out_operator)
        subgraphs.append(graph)

        op_num = len(block.ops)
        for i in range(op_num - 1, -1, -1):
            block._remove_op(i)
        # tmp_var = block.create_var(
        #     name="tmp", shape=[1], persistable=True, stop_gradient=True)
        if self.graph_idx == 0: # hack for startup program
            fetch_list = [block.var("learning_rate_0")]
        input_varlist = [var for var in input_varlist if var.is_data]
        
        for i in range(len(subgraphs)):
            block.append_op(
                type="ascend_trigger",
                inputs={"FeedList": input_varlist},
                outputs={"FetchList": fetch_list},
                attrs={'graph_idx': begin_graph_idx + i})
        self.graph_idx += len(subgraphs)
        return subgraphs

    def parse_program(self, startup_program, main_program, input_varlist, fetch_list):
        startup_subgraphs_with_id = []
        startup_subgraphs_with_id = self._parse_program("startup", startup_program)
        main_subgraphs_with_id = self._parse_program("main", main_program, input_varlist, fetch_list)
        return startup_subgraphs_with_id, main_subgraphs_with_id
        # return startup_subgraphs_with_id, []


def get_varlist(program):
    ret_list = []
    for var in program.list_vars():
        if var.is_data or var.persistable:
            ret_list.append(var)
    return ret_list

class AscendOptimizer(Optimizer):
    def __init__(self, optimizer, fetch_list=[]):
        self.inner_opt = optimizer
        self.fetch_list = fetch_list
        
    def __del__(self):
        core.ge_finalize()

    def _can_apply(self):
        if not self.user_defined_strategy.ascend:
            return False

        # TODO(hutuxian): other check here
        return True

    def _disable_strategy(self, dist_strategy):
        dist_strategy.ascend = False
        dist_strategy.ascend_configs = {}

    def minimize(self,
                      loss,
                      startup_program=None,
                      parameter_list=None,
                      no_grad_set=None):
        minimized = self.inner_opt.minimize(
            loss, startup_program=startup_program)


        self.ascend_instance = core.AscendInstance()
        #config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1"}
        config = {"ge.exec.deviceId": "0", "ge.graphRunMode": "1", "ge.exec.precision_mode": "must_keep_origin_dtype"} # "allow_mix_precision"}
        core.ge_initialize(config)
        self.ascend_instance.init_global_resources(
        )  # add whatever parameters here to init
        main_block = loss.block
        self.parser = AscendIRParser()

        input_varlist = get_varlist(main_block.program)
        print("input_varlist: ", input_varlist)
        startup_subgraphs_with_id, main_subgraphs_with_id = self.parser.parse_program(
            startup_program, main_block.program, input_varlist, self.fetch_list)
        idx = 0
        for graph_with_id in startup_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1
        for graph_with_id in main_subgraphs_with_id:
            self.ascend_instance.add_ascend_subgraph(idx, graph_with_id)
            idx += 1

        return minimized
