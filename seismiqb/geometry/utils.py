""" Useful functions, related to geometry. """


def time_to_depth(time, geometry):
    """ Convert time (in ms) into geometry depth value. """
    return round((time - geometry.delay) / geometry.sample_rate)

def depth_to_time(depth, geometry):
    """ Convert geometry depth value into time. """
    return round(depth * geometry.sample_rate + geometry.delay)
