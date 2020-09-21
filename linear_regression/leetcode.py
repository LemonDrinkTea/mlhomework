ans = []
an = []
he = 0
candidates=[2,3,5]
target=8
def dfs(begin):
    global he
    while (begin < len(candidates)):

        he += candidates[begin]

        if (he == target):
            an.append(candidates[begin])
            f = an[:]
            ans.append(f)
            an.pop()
            he -= candidates[begin]
            break
        if (he < target):
            an.append(candidates[begin])
            dfs(begin)
            if len(an):
                an.pop()

        he -= candidates[begin]
        begin += 1


dfs(0)
print(ans)



