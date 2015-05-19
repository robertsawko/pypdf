from numpy import array, abs
from statsmodels.nonparametric.kernel_density import KDEMultivariate


def box_to_particles(particles, x, a=1.0):
    ps = particles[abs(particles[:, 0] - x[0]) < a, :]
    ps = ps[abs(ps[:, 1] - x[1]) < a, :]
    ps = ps[abs(ps[:, 2] - x[2]) < a, :]
    return ps


def data_to_pdf(data, coords):
    num_of_variables = 1
    if len(data.shape) > 1:
        num_of_variables = data.shape[1]
    kde = KDEMultivariate(
        data=data, bw='normal_reference', var_type='c' * num_of_variables)
    return kde.pdf(coords)


def box_to_pdf(data, coords, x=array([0, 0, 0]), a=0.5):
    data_box = box_to_particles(data, x=x, a=a)
    data_box = data_box[:, 3]
    pdf_box = data_to_pdf(data_box, coords)
    return pdf_box
