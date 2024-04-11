from ir import *
from asg import *
from helpers import rebind_iterate


def _get_val(ir):
    if type(ir) == Literal:
        return ir.val
    elif type(ir) in (int, float):
        return ir
    else:
        return None

def _is_full_slicing(ir):
    if type(ir) == Indexing and type(ir.idx) == Indexing and type(ir.idx.dobject) == Slice and type(ir.idx.idx) == Literal and ir.idx.idx.val == -1:
        if type(ir.dobject) == Ndarray:
            s = _get_val(ir.dobject.size[0])
        elif type(ir.dobject) == Indexing:
            s = _get_val(ir.dobject.size[ir.dobject.ref_point])
        else:
            s = None
        if _get_val(ir.idx.dobject.start) == 0 and _get_val(ir.idx.dobject.step) == 1 and s != None and _get_val(ir.idx.dobject.stop.val) == s:
            return True

def _remove_full_slicing(ir):
    if type(ir) == list or type(ir) == tuple:
        for l in ir:
            _remove_full_slicing(l)
    elif type(ir) == Loop:
        _remove_full_slicing(ir.body)
    elif type(ir) == Expr:
        if _is_full_slicing(ir.left):
            _remove_full_slicing(ir.left.dobject)
            ir.left = ir.left.dobject
        else:
            _remove_full_slicing(ir.left)
        if _is_full_slicing(ir.right):
            _remove_full_slicing(ir.right.dobject)
            ir.right = ir.right.dobject
        else:
            _remove_full_slicing(ir.right)
    elif type(ir) == Assignment:
        if _is_full_slicing(ir.lhs):
            _remove_full_slicing(ir.lhs.dobject)
            ir.lhs = ir.lhs.dobject
        else:
            _remove_full_slicing(ir.lhs)
        if _is_full_slicing(ir.rhs):
            _remove_full_slicing(ir.rhs.dobject)
            ir.rhs = ir.rhs.dobject
        else:
            _remove_full_slicing(ir.rhs)
    elif type(ir) == Indexing:
        if _is_full_slicing(ir.dobject):
            _remove_full_slicing(ir.dobject.dobject)
            ir.dobject = ir.dobject.dobject
        else:
            _remove_full_slicing(ir.dobject)
        if _is_full_slicing(ir.idx):
            _remove_full_slicing(ir.idx)
            ir.idx = ir.idx.dobject
        else:
            _remove_full_slicing(ir.idx)
    elif type(ir) == Slice:
        if _is_full_slicing(ir.start):
            _remove_full_slicing(ir.start.dobject)
            ir.start = ir.start.dobject
        else:
            _remove_full_slicing(ir.start)
        if _is_full_slicing(ir.stop):
            _remove_full_slicing(ir.stop.dobject)
            ir.stop = ir.stop.dobject
        else:
            _remove_full_slicing(ir.stop)
        if _is_full_slicing(ir.step):
            _remove_full_slicing(ir.step.dobject)
            ir.step = ir.step.dobject
        else:
            _remove_full_slicing(ir.step)
    elif type(ir) == Math:
        if _is_full_slicing(ir.val):
            _remove_full_slicing(ir.dobject)
            ir.val = ir.dobject
        else:
            _remove_full_slicing(ir.val)






def _tensorize_loops(scope):
    for stmt in scope[:]:
        if type(stmt) == Loop:
            _tensorize_loops(stmt.body)
            rebind_iterate(stmt.body, stmt.iterate, Indexing(Slice(stmt.start, stmt.end, stmt.step), Literal(-1, 'int')))
            _remove_full_slicing(stmt.body)
            idx = scope.index(stmt)
            scope[idx:idx] = stmt.body
            scope.remove(stmt)



def tensorize(asg):
    def action(node, res):
        if type(node) == TensorOp:
            res.append(node.compute)

    t = helpers.Traversal(action)
    ir = t(asg)
    for l in ir:
        _tensorize_loops(l)