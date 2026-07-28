/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.knn.plugin;

import org.opensearch.index.mapper.DynamicFieldTypeInferencer;
import org.opensearch.index.mapper.DynamicValueSummary;
import org.opensearch.index.mapper.FieldValueParserSupplier;
import org.opensearch.knn.index.mapper.KNNVectorFieldMapper;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * k-NN implementation of {@link DynamicFieldTypeInferencer}.
 *
 * <p>Claims unmapped fields whose value is a flat numeric array with at least
 * {@link #MIN_VECTOR_DIMENSION} elements whose count is a multiple of 8, and maps them as
 * {@code knn_vector}. The dimension is inferred from the array length of the first document —
 * subsequent documents with a different dimension are rejected by the mapper.
 *
 * <p>Core classifies the buffered value ({@link DynamicValueSummary}) and passes the shape + array
 * length, so this inferencer no longer streams the tokens itself to answer "is this a flat numeric
 * array." It only applies the k-NN gate (≥ {@link #MIN_VECTOR_DIMENSION} and a multiple of 8) — a
 * plugin-side policy that can be relaxed without any core change. Below-threshold or non-numeric
 * arrays are declined (return {@code null}), so they fall through to the normal float path — the
 * gated "middle path": core classifies, but k-NN does not claim every numeric array.
 */
public class KNNDynamicFieldTypeInferencer implements DynamicFieldTypeInferencer {

    static final int MIN_VECTOR_DIMENSION = 128;

    /**
     * Returns a knn_vector mapping config when core reports a flat numeric array whose length is at
     * least {@link #MIN_VECTOR_DIMENSION} and a multiple of 8; otherwise {@code null} to pass.
     *
     * @param summary core's classification of the value shape and array length
     * @param fieldValueParser unused here — the summary already carries everything k-NN needs
     * @return mutable config map {@code {type: knn_vector, dimension: N}} if claimed, or {@code null} to pass
     */
    @Override
    public Map<String, Object> inferFieldType(DynamicValueSummary summary, FieldValueParserSupplier fieldValueParser)
        throws IOException {
        if (summary.isFlatNumericArray() == false) {
            return null;
        }
        int count = summary.arrayLength();
        if (count < MIN_VECTOR_DIMENSION || count % 8 != 0) {
            return null;
        }
        Map<String, Object> config = new HashMap<>();
        config.put("type", KNNVectorFieldMapper.CONTENT_TYPE);
        config.put("dimension", count);
        return config;
    }
}
