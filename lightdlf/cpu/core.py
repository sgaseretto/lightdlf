import numpy as np


class Tensor(object):

    def __init__(self, data,
                 autograd=False,
                 creators=None,
                 creation_op=None,
                 id=None):
        '''
        Inicializa un tensor utilizando numpy

        @data: una lista de numeros
        @creators: lista de tensores que participarion en la creacion de un nuevo tensor
        @creators_op: la operacion utilizada para combinar los tensores en el nuevo tensor
        @autograd: determina si se realizara backprop o no sobre el tensor
        @id: identificador del tensor, para poder dar seguimiento a los hijos y padres del mismo
        '''
        self.data = np.array(data)
        self.creation_op = creation_op
        self.creators = creators
        self.grad = None
        self.autograd = autograd
        self.children = {}
        # se asigna un id al tensor
        if (id is None):
            id = np.random.randint(0, 100000)
        self.id = id

        # se hace un seguimiento de cuantos hijos tiene un tensor
        # si los creadores no es none
        if (creators is not None):
            # para cada tensor padre
            for c in creators:
                # se verifica si el tensor padre posee el id del tensor hijo
                # en caso de no estar, agrega el id del tensor hijo al tensor padre
                if (self.id not in c.children):
                    c.children[self.id] = 1
                # si el tensor ya se encuentra entre los hijos del padre
                # y vuelve a aparece, se incrementa en uno
                # la cantidad de apariciones del tensor hijo
                else:
                    c.children[self.id] += 1

    def all_children_grads_accounted_for(self):
        '''
        Verifica si un tensor ha recibido la cantidad
        correcta de gradientes por cada uno de sus hijos
        '''
        # print('tensor id:', self.id)
        for id, cnt in self.children.items():
            if (cnt != 0):
                return False
        return True

    def backward(self, grad, grad_origin=None):
        '''
        Funcion que propaga recursivamente el gradiente a los creators o padres del tensor

        @grad: gradiente
        @grad_orign
        '''
        if (self.autograd):
            if grad is None:
                grad = Tensor(np.ones_like(self.data))
            if (grad_origin is not None):
                # Verifica para asegurar si se puede hacer retropropagacion
                if (self.children[grad_origin.id] == 0):
                    raise Exception("No se puede retropropagar mas de una vez")
                # o si se está esperando un gradiente, en dicho caso se decrementa
                else:
                    # el contador para ese hijo
                    self.children[grad_origin.id] -= 1

        # acumula el gradiente de multiples hijos
        if (self.grad is None):
            self.grad = grad
        else:
            self.grad += grad

        if (self.creators is not None and
                (self.all_children_grads_accounted_for() or grad_origin is None)):

            if (self.creation_op == 'neg'):
                self.creators[0].backward(self.grad.__neg__())

            if (self.creation_op == 'add'):
                # al recibir self.grad, empieza a realizar backprop
                self.creators[0].backward(self.grad, grad_origin=self)
                self.creators[1].backward(self.grad, grad_origin=self)

            if (self.creation_op == "sub"):
                self.creators[0].backward(Tensor(self.grad.data), self)
                self.creators[1].backward(Tensor(self.grad.__neg__().data), self)

            if (self.creation_op == "mul"):
                new = self.grad * self.creators[1]
                self.creators[0].backward(new, self)
                new = self.grad * self.creators[0]
                self.creators[1].backward(new, self)

            if (self.creation_op == "mm"):
                layer = self.creators[0]  # activaciones => layer
                weights = self.creators[1]  # pesos = weights
                # c0 = self.creators[0]                       # activaciones => layer
                # c1 = self.creators[1]                       # pesos = weights
                # new = self.grad.mm(c1.transpose())  # grad = delta => delta x weights.T
                new = Tensor.mm(self.grad, weights.transpose())  # grad = delta => delta x weights.T
                layer.backward(new)
                # c0.backward(new)
                # new = self.grad.transpose().mm(c0).transpose() # (delta.T x layer).T = layer.T x delta
                new = Tensor.mm(layer.transpose(), self.grad)  # layer.T x delta
                weights.backward(new)
                # c1.backward(new)

            if (self.creation_op == "transpose"):
                self.creators[0].backward(self.grad.transpose())

            if ("sum" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.expand(dim, self.creators[0].data.shape[dim]))

            if ("expand" in self.creation_op):
                dim = int(self.creation_op.split("_")[1])
                self.creators[0].backward(self.grad.sum(dim))

            if (self.creation_op == "index_select"):
                # new_grad es un vector de 0s que luego contendra
                # los gradientes para cada embedding
                new_grad = np.zeros_like(self.creators[0].data)
                # se obtienen los indices en un vector unidimensional
                indices_ = self.index_select_indices.data.flatten()
                grad_ = grad.data.reshape(len(indices_), -1)
                for i in range(len(indices_)):
                    new_grad[indices_[i]] = new_grad[indices_[i]] + grad_[i]
                self.creators[0].backward(Tensor(new_grad))

            if (self.creation_op == "sigmoid"):
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (self * (ones - self)))

            if (self.creation_op == "tanh"):
                ones = Tensor(np.ones_like(self.grad.data))
                self.creators[0].backward(self.grad * (ones - (self * self)))

            if (self.creation_op == 'relu'):
                mask = Tensor(self.data > 0)
                self.creators[0].backward(self.grad * mask)

            if (self.creation_op == "cross_entropy"):
                dx = self.softmax_output - self.target_dist
                self.creators[0].backward(Tensor(dx))

    def __neg__(self):
        if (self.autograd):
            return Tensor(self.data * -1,
                          autograd=True,
                          creators=[self],
                          creation_op='neg')
        return Tensor(self.data * -1)

    def __add__(self, other):
        '''
        @other: un Tensor
        '''
        if (self.autograd and other.autograd):
            return Tensor(self.data + other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='add')
        return Tensor(self.data + other.data)

    def __sub__(self, other):
        '''
        @other: un Tensor
        '''
        if (self.autograd and other.autograd):
            return Tensor(self.data - other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op='sub')
        return Tensor(self.data - other.data)

    def __mul__(self, other):
        '''
        @other: un Tensor
        '''
        if (self.autograd and other.autograd):
            return Tensor(self.data * other.data,
                          autograd=True,
                          creators=[self, other],
                          creation_op="mul")
        return Tensor(self.data * other.data)

    def sum(self, dim):
        '''
        Suma atravez de dimensiones, si tenemos una matriz 2x3 y
        aplicamos sum(0) sumara todos los valores de las filas
        dando como resultado un vector 1x3, en cambio si se aplica
        sum(1) el resultado es un vector 2x1

        @dim: dimension para la suma
        '''
        if (self.autograd):
            return Tensor(self.data.sum(dim),
                          autograd=True,
                          creators=[self],
                          creation_op="sum_" + str(dim))
        return Tensor(self.data.sum(dim))

    def expand(self, dim, copies):
        '''
        Se utiliza para retropropagar a traves de una suma sum().
        Copia datos a lo largo de una dimension
        '''

        trans_cmd = list(range(0, len(self.data.shape)))
        trans_cmd.insert(dim, len(self.data.shape))
        new_data = self.data.repeat(copies).reshape(list(self.data.shape) + [copies]).transpose(trans_cmd)

        if (self.autograd):
            return Tensor(new_data,
                          autograd=True,
                          creators=[self],
                          creation_op="expand_" + str(dim))
        return Tensor(new_data)

    def transpose(self):
        if (self.autograd):
            return Tensor(self.data.transpose(),
                          autograd=True,
                          creators=[self],
                          creation_op="transpose")

        return Tensor(self.data.transpose())

    def mm(self, x):
        if (self.autograd):
            return Tensor(self.data.dot(x.data),
                          autograd=True,
                          creators=[self, x],
                          creation_op="mm")
        return Tensor(self.data.dot(x.data))

    def index_select(self, indices):
        if (self.autograd):
            new = Tensor(self.data[indices.data],
                         autograd=True,
                         creators=[self],
                         creation_op='index_select')
            # se agregan los indices como atributos del tensor
            new.index_select_indices = indices
            return new
        return Tensor(self.data[indices.data])

    def sigmoid(self):
        if (self.autograd):
            return Tensor(1 / (1 + np.exp(-self.data)),
                          autograd=True,
                          creators=[self], creation_op='sigmoid')
        return Tensor(1 / (1 + np.exp(-self.data)))

    def tanh(self):
        if (self.autograd):
            return Tensor(np.tanh(self.data),
                          autograd=True,
                          creators=[self],
                          creation_op='tanh')
        return Tensor(np.tanh(self.data))

    def relu(self):
        ones_and_zeros = self.data > 0
        if (self.autograd):
            return Tensor(self.data * ones_and_zeros,
                          autograd=True,
                          creators=[self],
                          creation_op='relu')
        return Tensor(self.data * ones_and_zeros)

    def cross_entropy(self, target_indices):
        temp = np.exp(self.data)
        softmax_output = temp / np.sum(temp,
                                       axis=len(self.data.shape) - 1,
                                       keepdims=True)

        t = target_indices.data.flatten()
        p = softmax_output.reshape(len(t), -1)
        target_dist = np.eye(p.shape[1])[t]
        loss = -(np.log(p) * (target_dist)).sum(1).mean()

        if (self.autograd):
            out = Tensor(loss,
                         autograd=True,
                         creators=[self],
                         creation_op='cross_entropy')
            out.softmax_output = softmax_output
            out.target_dist = target_dist
            return out
        return Tensor(loss)

    def __repr__(self):
        return str(self.data.__repr__())

    def __str__(self):
        return str(self.data.__str__())
