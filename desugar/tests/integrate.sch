


; Home spun foldl which allows us to employ and test additional user-land higher-order functions
(define (myfold foldf acc lst)
        (if (null? lst)
            acc
            (myfold foldf (foldf (car lst) acc) (cdr lst))))


; A function for numerical integration over fun: Num -> Num between start and stop with delta precision/step using a quadrature rule
(define (integrate fun rule start stop delta) 
        (letrec ([generatepoints (lambda (pointslst)
                (let ([last (car (car pointslst))])
                     (if (< last start)
                         (cdr pointslst)
                         (generatepoints (cons (cons (- last delta) last) pointslst)))))])
        (myfold (lambda (pnts sum) (+ sum (rule fun pnts))) 0 (generatepoints (list (cons (- stop delta) stop))))))


; a function taking a function, pair of points, and a delta, which returns the area of that slice by the midpoint rule
(define (midpoint_rule func points)
        (* (- (cdr points) (car points))
           (func (/ (+ (car points) (cdr points)) 2.0))))


; a quadrature rule for calculating the area of a slice as a trapazoid
(define (trapezoidal_rule func points)
        (let ([a (func (car points))]
              [b (func (cdr points))])
             (let ([s (min a b)]
                   [b (max a b)])
                  (* (- (cdr points) (car points))
                     (+ s (/ (- b s) 2.0))))))


; a rule for overapproximating based on the largest of two points
(define (biggest_rule func points)
        (let ([a (func (car points))]
              [b (func (cdr points))])
             (let ([b (max a b)])
                  (* (- (cdr points) (car points)) b))))


; a rule for underapproximating based on the smallest of two points
(define (smallest_rule func points)
        (let ([a (func (car points))]
              [b (func (cdr points))])
             (let ([s (min a b)])
                  (* (- (cdr points) (car points)) s))))


; All rules
(define rules (list midpoint_rule trapezoidal_rule biggest_rule smallest_rule))


; Some functions to integrate over
(define funcs (list (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))(lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))(lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))(lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))(lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    (lambda (x) (sqrt x))
                    (lambda (x) (* x x))
                    (lambda (x) (* x x x))
                    (lambda (x) (* x x 2))
                    (lambda (x) (* x x 3))
                    (lambda (x) (* x 2))
                    (lambda (x) (* x 3))
                    (lambda (x) (+ x 1))
                    (lambda (x) (+ x 2))
                    (lambda (x) (/ x 3.0))
                    (lambda (x) (expt x x))
                    (lambda (x) (- (expt x x) (* x x)))
                    (lambda (x) 2)
                    (lambda (x) (+ (* x x) x))
                    (lambda (x) (- (* x x) x))
                    (lambda (x) (/ x (* x x)))
                    (lambda (x) (max (* x x) (sqrt x)))
                    (lambda (x) (min (* x x) (sqrt x)))
                    ))


(myfold (lambda (rule na) (myfold (lambda (afunc na) (pretty-print (integrate afunc rule 0.0 2.0 0.001))) na funcs)) (void) rules)

