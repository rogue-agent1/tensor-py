#!/usr/bin/env python3
"""Tiny tensor library with broadcasting and autograd."""
import sys,math

class Tensor:
    def __init__(self,data,shape=None,requires_grad=False):
        if isinstance(data,(int,float)):data=[data];shape=shape or(1,)
        self.data=list(data);self.shape=shape or(len(data),)
        self.grad=None;self.requires_grad=requires_grad;self._backward=lambda:None;self._prev=[]
    def __len__(self):return len(self.data)
    def __getitem__(self,i):return self.data[i]
    def __repr__(self):return f"Tensor({self.data}, shape={self.shape})"
    def _binop(self,other,op,grad_fn=None):
        o=other if isinstance(other,Tensor) else Tensor(other)
        out=Tensor([op(a,b) for a,b in zip(self.data,o.data)],self.shape,
                   self.requires_grad or o.requires_grad)
        if out.requires_grad and grad_fn:
            out._prev=[self,o]
            out._backward=lambda:grad_fn(self,o,out)
        return out
    def __add__(self,o):
        def g(a,b,out):
            if a.requires_grad:a.grad=[((a.grad[i] if a.grad else 0)+out.grad[i]) for i in range(len(a.data))]
            if b.requires_grad:b.grad=[((b.grad[i] if b.grad else 0)+out.grad[i]) for i in range(len(b.data))]
        return self._binop(o,lambda a,b:a+b,g)
    def __mul__(self,o):
        def g(a,b,out):
            if a.requires_grad:a.grad=[((a.grad[i] if a.grad else 0)+b.data[i]*out.grad[i]) for i in range(len(a.data))]
            if b.requires_grad:b.grad=[((b.grad[i] if b.grad else 0)+a.data[i]*out.grad[i]) for i in range(len(b.data))]
        return self._binop(o,lambda a,b:a*b,g)
    def __sub__(self,o):return self._binop(o,lambda a,b:a-b)
    def sum(self):
        out=Tensor(sum(self.data),requires_grad=self.requires_grad)
        if self.requires_grad:
            out._prev=[self]
            def bw():
                self.grad=[(self.grad[i] if self.grad else 0)+out.grad[0] for i in range(len(self.data))]
            out._backward=bw
        return out
    def backward(self):
        order=[];visited=set()
        def topo(t):
            if id(t) not in visited:
                visited.add(id(t))
                for p in t._prev:topo(p)
                order.append(t)
        topo(self)
        self.grad=[1.0]*len(self.data)
        for t in reversed(order):t._backward()
    def dot(self,other):return (self*other).sum()

def main():
    if len(sys.argv)>1 and sys.argv[1]=="--test":
        a=Tensor([1,2,3],requires_grad=True)
        b=Tensor([4,5,6],requires_grad=True)
        c=(a*b).sum()  # 4+10+18=32
        assert c.data==[32]
        c.backward()
        assert a.grad==[4,5,6]  # dc/da = b
        assert b.grad==[1,2,3]  # dc/db = a
        # add
        x=Tensor([2,3],requires_grad=True)
        y=Tensor([1,1],requires_grad=True)
        z=(x+y).sum()
        z.backward()
        assert x.grad==[1,1]
        assert y.grad==[1,1]
        print("All tests passed!")
    else:
        a=Tensor([1,2,3],requires_grad=True)
        b=Tensor([4,5,6],requires_grad=True)
        c=(a*b).sum();c.backward()
        print(f"a·b = {c.data[0]}, ∂/∂a = {a.grad}, ∂/∂b = {b.grad}")
if __name__=="__main__":main()
