from functools import total_ordering

# how many nearest neighbors to consider for calculating novelty score?
KNN = 15
# the maximal novelty archive size
MAXNoveltyArchiveSize = 1000


@total_ordering
class NoveltyItem:

    def __init__(self, generation=-1, genomeId=-1, novelty=-1, data=[]):
        self.generation = generation
        self.genomeId = genomeId
        self.novelty = novelty
        self.data = data

    def __lt__(self, other):
        if not self._is_valid_operand(other):
            return NotImplemented
        return self.novelty < other.novelty

    def __str__(self):
        return "%s: id: %d, at generation: %d, novelty: %f\tdata: %s" % \
            (self.__class__.__name__, self.genomeId, self.generation, self.novelty, self.data)

    def _is_valid_operand(self, other):
        return (hasattr(other, "novelty"))


class NoveltyArchive:

    def __init__(self, metric=None, max_novelty_size=None) -> None:
        self.max_novelty_size = max_novelty_size or MAXNoveltyArchiveSize
        self.novelty_metric = metric or self._default_metric
        self.novel_items = []

    def __len__(self):
        return len(self.novel_items)

    def __str__(self):
        return f'NoveltyArchive: {len(self)} items'

    def evaluate_novelty_score(self, nitem, nitems) -> float:
        # check against current population
        distances = [
            self.novelty_metric(nitem, x)
            for x in nitems
            if x.genomeId != nitem.genomeId
        ]
        # check against other novelty items
        distances += [
            self.novelty_metric(nitem, x)
            for x in self.novel_items
            if x.genomeId != nitem.genomeId
        ]

        distances.sort()
        nitem.novelty = (sum(distances[:KNN]) / KNN) / len(nitem.data)#**2
        self._add_novelty_item(nitem)
        return nitem.novelty

    def _add_novelty_item(self, item):
        if len(self) >= self.max_novelty_size:
            if item > self.novel_items[-1]:
                self.novel_items[-1] = item
        else:
            self.novel_items.append(item)
            # item.in_archive = True
        self.novel_items.sort(reverse=True)

    def _default_metric(self, first, second):
        return len(set(first.data) - set(second.data))


if __name__ == "__main__":
    a = NoveltyItem(1, 1, data=['hoped', 'hogan', 'suite', 'arett', 'avion', 'lotsa'])
    b = NoveltyItem(1, 2, data=['hoped', 'hogan', 'suite', 'arett', 'avion', 'lotsa'])
    c = NoveltyItem(2, 3, data=['hoped', 'hogan', 'suite', 'arett', 'avion', 'tares'])
    d = NoveltyItem(2, 4, data=['hoped', 'hogan', 'suite', 'arett', 'asdfg', 'tares'])
    e = NoveltyItem(2, 5, data=['asdf', 'fdsa'])
    f = NoveltyItem(3, 6, data=['asdf', 'fdsa', 'qwerty', 'ytrewq', 'zxcvbn', 'nbvcxz', 'edxc', 'qwert', 'poiun'])
    items = [a, b, c, d, e]
    print(a)
    print(b)
    print(c)
    print(d)
    print(e)
    print(f)

    # def metric_test(first, second):
    #     return 8675309

    print('---')
    arch = NoveltyArchive(max_novelty_size=2)
    print(f'a: {arch.evaluate_novelty_score(a, items)}') # 0.09
    print(f'b: {arch.evaluate_novelty_score(b, items)}') # 0.09
    print(f'c: {arch.evaluate_novelty_score(c, items)}') # 0.12
    print(f'd: {arch.evaluate_novelty_score(d, items)}') # 0.15
    print(f'e: {arch.evaluate_novelty_score(e, items)}') # 0.4
    print(f'f: {arch.evaluate_novelty_score(f, items)}') # 0.4

    print('---')
    for x in arch.novel_items:
        print(x)
