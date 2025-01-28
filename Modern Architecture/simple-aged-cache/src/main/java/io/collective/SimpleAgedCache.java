package io.collective;
import java.time.Instant;
import java.util.Map;
import java.time.Clock;
import java.util.HashMap;
public class SimpleAgedCache {
    // declare vars
    private final Clock clock;
    private final Map<Object,Object[]> cacheTest;

    public SimpleAgedCache(Clock clock) {
        this.clock=clock;
        // initiate cache
        this.cacheTest=new HashMap<>();
    }

    public SimpleAgedCache() {
        //if no clock is given
        this(Clock.systemDefaultZone());
    }

    public void put(Object key, Object value, int retentionInMillis) {
        // calculate expiration time
        Instant exp=clock.instant().plusMillis(retentionInMillis);
        // put cache
        cacheTest.put(key, new Object[] { value, exp });
    }

    public boolean isEmpty() {
        if (cacheTest.isEmpty()) {
            return true;
        }
        return false;
    }

    public int size() {

        // loop through all entries and manually kill cache
        for (Map.Entry<Object,Object[]> ent: cacheTest.entrySet() ) {
            Object k=ent.getKey();
            Object[] val=ent.getValue();
            //Object val0=val[0];
            Instant val1= (Instant)  val[1];
            if (val1.isBefore(Instant.now(clock))) {
                cacheTest.remove(k);
            }
        }

        return cacheTest.size();
    }

    public Object get(Object key) {

        Object[] val=cacheTest.get(key);
        if (val==null) {
            return null;
        } else {
            Object val0=val[0];
            Instant val1= (Instant)  val[1];
            if (val1.isBefore(Instant.now(clock))) {
                cacheTest.remove(key);
                return null;
            } else {
                return val0;
            }
        }
    }
}