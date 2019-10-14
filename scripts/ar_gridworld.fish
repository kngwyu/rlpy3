for map in {6x6guided, 6x6sparse, 6x6liar, 6x6spelunky, 6x6speguided}
    pipenv run python ../examples/ar_gridworld.py \
            --agent=tabular-q \
            --log-dir="Result/$map-ep01" \
            --map=$map \
            --num-policy-checks=20 \
            --max-steps=4000 \
            --epsilon=0.1 \
            train \
            --plot-save
    if test $status != 0
        echo $map "failed"
        exit
    end
end