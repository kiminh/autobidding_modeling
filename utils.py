import tensorflow as tf


def tf_print(tensor, message=''):
    """
    print a tensor for debugging
    """
    def _print_tensor(tensor):
        print(message, tensor)
        return tensor

    log_op = tf.py_func(_print_tensor, [tensor], [tensor.dtype])[0]
    with tf.control_dependencies([log_op]):
        res = tf.identity(tensor)

    return res


